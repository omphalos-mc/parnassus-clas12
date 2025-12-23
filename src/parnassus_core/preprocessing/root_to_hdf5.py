import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final

import h5py
import hdf5plugin
import numpy as np
import uproot
from numpy.lib.recfunctions import unstructured_to_structured
from tqdm import tqdm

MAX_PARTICLES = 400


def get_parser():
    """
    Argument parser for Preprocessing script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(description="Converting command line options.")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Enter the name of the input file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default=None,
        help="Enter the name of the output file. If not provided, the input name is used (with .h5 extension).",
    )

    parser.add_argument(
        "-n",
        "--nevents",
        type=int,
        required=False,
        default=None,
        help="Enter the number of events to be processed.",
    )

    parser.add_argument(
        "-m",
        "--max_particles",
        type=int,
        required=False,
        default=MAX_PARTICLES,
        help="Enter the maximum number of particles per event.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=10_000,
        help="Enter the batch size for processing events. Used to write data in chunks.",
    )

    return parser.parse_args()


@dataclass
class TreeMap:
    # Define the structure for mapping tree variables
    tree_name: str
    global_vars: list[str]  # e.g., ['eventNumber', ...]
    constituents: dict[str, list[str]]  # e.g., {'tracks': [...], 'cells': [...]}

    @classmethod
    def create(cls, tree_name: str, tree_keys: list[str]) -> "TreeMap":
        constituents = {}
        global_vars = []
        for key in tree_keys:
            if "event" in key.lower():
                global_vars.append(key)
                continue
            if key in {"npflow", "ntruth"}:
                continue
            constituent_name, var_name = key.split("_", 1)
            if constituent_name not in constituents:
                constituents[constituent_name] = []
            constituents[constituent_name].append(var_name)
        return cls(
            tree_name=tree_name, global_vars=global_vars, constituents=constituents
        )

    def __repr__(self) -> str:
        return (
            f"TreeMap(tree_name={self.tree_name}, "
            f"global_vars={self.global_vars}, "
            f"constituents={self.constituents})"
        )


@final
class PrepareSamples:
    """
    This class is preparing the samples for further processing defined in the
    configuration file:
        - writes these iteratively to h5 output files

    This class will take the information provided in the `samples` block in
    the preprocessing config.
    """

    def __init__(self, args) -> None:
        """Preparation of h5 samples.

        Parameters
        ----------
        args : parse_args
            command line arguments
        config : config class
            preprocessing configuration class containing all info about preprocessing
        """
        self.input = Path(args.input)
        if args.output is None:
            self.output = self.input.with_suffix(".h5")
        else:
            self.output = Path(args.output)

        self.n_events = args.nevents
        self.max_particles = args.max_particles
        self.batch_size = args.batch_size

        self.data_map, total_events = self.get_metadata(self.input)
        if self.n_events is None or self.n_events > total_events:
            self.n_events = total_events

    def get_metadata(self, filename: str | Path) -> tuple[TreeMap, int]:
        """
        Get the data map from the ROOT file.

        Parameters
        ----------
        filename : str | Path
            Path to the ROOT file.
        Returns
        -------
        tuple[TreeMap, int]
            The TreeMap object and the total number of entries in the tree.
        """
        with uproot.open(filename) as rootfile:  # type: ignore
            tree = rootfile.keys()[0]  # get first tree
            tree_obj = rootfile[tree]
            tree_keys = tree_obj.keys()
            return TreeMap.create(tree, tree_keys), tree_obj.num_entries

    def _load_events(
        self, filename: str | Path, start: int, end: int
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        with uproot.open(filename) as rootfile:
            tree_name = self.data_map.tree_name
            tree = rootfile[tree_name]
            arrays = tree.arrays(
                entry_start=start,
                entry_stop=end,
                library="np",
            )
            tree_data = {}
            for global_var in self.data_map.global_vars:
                tree_data[global_var] = arrays[global_var]
            for constituent_name, var_names in self.data_map.constituents.items():
                constituent_arrays = {
                    var_name: arrays[f"{constituent_name}_{var_name}"]
                    for var_name in var_names
                }
                tree_data[constituent_name] = constituent_arrays
        return tree_data

    def _convert_events(self, events):
        event_dtypes = []
        events_output = []
        for key in events.keys():
            event_dtypes.append((key, "f4"))
            events_output.append(events[key].astype(np.float32))
        return unstructured_to_structured(
            np.stack(events_output, axis=-1), dtype=np.dtype(event_dtypes)
        )

    def _convert_constituents(self, arrays, max_size, start_index=0, end_index=None):
        dtypes = []
        output = []
        # sort by pt if available, otherwise by energy
        if "pt" in arrays:
            sort_idxs = [
                np.argsort(-arrays["pt"][i])
                for i in range(
                    start_index, end_index if end_index else arrays["pt"].shape[0]
                )
            ]
        else:
            sort_idxs = [
                np.argsort(-arrays["e"][i])
                for i in range(
                    start_index, end_index if end_index else arrays["e"].shape[0]
                )
            ]
        for key, arr in arrays.items():
            # We pad with NaNs to indicate invalid entries
            # shape: (n_events, max_size)
            # Also we sort the entries by pt/energy
            output.append(
                np.array(
                    [
                        np.pad(
                            arr[i][sort_idxs[i - start_index]][:max_size].astype(
                                np.float32
                            ),
                            (0, max_size - len(arr[i][:max_size])),
                            "constant",
                            constant_values=np.nan if key != "class" else 0,
                        )
                        if arr[i].size > 0
                        else [np.nan for _ in range(max_size)]
                        for i in range(
                            start_index, end_index if end_index else arr.shape[0]
                        )
                    ]
                )
            )
            dtypes.append((key, "f4"))
        # create valid mask
        output.append(np.isfinite(output[0]) * 1)
        dtypes.append(("valid", "bool"))
        return unstructured_to_structured(
            np.stack(output, axis=-1), dtype=np.dtype(dtypes)
        )

    def _compute_event_features(self, arrays: np.ndarray) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        data_dict["ht"] = np.nansum(arrays["pt"], axis=1)
        data_dict["met_x"] = np.nansum(arrays["pt"] * np.cos(arrays["phi"]), axis=1)
        data_dict["met_y"] = np.nansum(arrays["pt"] * np.sin(arrays["phi"]), axis=1)
        return data_dict

    def run(self):
        """Run the preprocessing to convert ROOT to HDF5."""
        # load data
        print("Loading data...")
        tree_data = self._load_events(self.input, 0, self.n_events)
        print("Converting and writing data...")
        batch_indices = [
            (i, min(i + self.batch_size, self.n_events))
            for i in range(0, self.n_events, self.batch_size)
        ]
        create_file = True
        displayed_writing_output = False
        pbar = tqdm(total=self.n_events)
        for start, end in batch_indices:
            h5_tree_data = {}
            event_data_dict = {
                k: tree_data[k][start:end] for k in self.data_map.global_vars
            }
            # convert constituents
            for constituent_name in self.data_map.constituents:
                h5_tree_data[constituent_name] = self._convert_constituents(
                    tree_data[constituent_name],
                    self.max_particles,
                    start_index=start,
                    end_index=end,
                )
                for key, value in self._compute_event_features(
                    h5_tree_data[constituent_name]
                ).items():
                    event_data_dict[f"{constituent_name}_{key}"] = value
            # convert global event variables
            h5_tree_data["events"] = self._convert_events(event_data_dict)
            pbar.update(end - start)
            if create_file:
                create_file = False
                # write to file by creating dataset
                pbar.write(f"Creating output file: {self.output}")
                displayed_writing_output = True
                with h5py.File(self.output, "w") as out_file:
                    for key, value in h5_tree_data.items():
                        out_file.create_dataset(
                            key,
                            data=value,
                            maxshape=(None,)
                            if key == "events"
                            else (None, self.max_particles),
                            chunks=True,
                            compression=hdf5plugin.LZ4(),
                        )

            else:
                # appending to existing dataset
                if displayed_writing_output:
                    pbar.write(f"Writing to output file: {self.output}")
                with h5py.File(self.output, "a") as out_file:
                    for key, value in h5_tree_data.items():
                        dataset = cast(h5py.Dataset, out_file[key])
                        dataset.resize((dataset.shape[0] + value.shape[0]), axis=0)
                        dataset[-value.shape[0] :] = value
                displayed_writing_output = False
        pbar.close()

    def validate(self, batch_size: int = 100) -> bool:
        """Validate the output file by comparing random samples from HDF5 and ROOT files.

        Parameters
        ----------
        batch_size : int
            Number of sequential events to read and validate at once (default: 100)

        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        import random

        # Check if output file exists
        if not self.output.exists():
            raise FileNotFoundError(f"Output file {self.output} does not exist")

        # Get total number of events in HDF5 file
        with h5py.File(self.output, "r") as h5file:
            n_events_h5 = h5file["events"].shape[0]

        # Calculate number of batches and select random starting positions
        max_start = max(0, n_events_h5 - batch_size)
        random_start = random.sample(range(max_start + 1), 1)[0]

        start_index = random_start
        end_index = min(random_start + batch_size - 1, n_events_h5)

        print(
            f"Validating {batch_size} events from {n_events_h5} total events starting at {random_start}..."
        )

        all_match = True

        # Process each batch
        # Load ROOT data for this batch range
        tree_data = self._load_events(self.input, start_index, end_index)

        # Load HDF5 data for this batch
        with h5py.File(self.output, "r") as h5file:
            h5_events = cast(h5py.Dataset, h5file["events"][start_index:end_index])
            h5_constituents = {
                name: cast(h5py.Dataset, h5file[name][start_index:end_index])
                for name in self.data_map.constituents
            }

        # Compare global event variables
        for global_var in self.data_map.global_vars:
            root_values = tree_data[global_var].astype(np.float32)
            h5_values = h5_events[global_var]

            if not np.allclose(
                root_values, h5_values, equal_nan=True, rtol=1e-6, atol=1e-8
            ):
                print(f"  events.{global_var}: Mismatch detected")
                all_match = False

        # Compare constituents using the valid field
        for constituent_name in self.data_map.constituents:
            h5_const = h5_constituents[constituent_name]

            # Get valid mask from HDF5
            valid_mask = h5_const["valid"]
            root_idxs = [
                np.argsort(-tree_data[constituent_name]["pt"][i])
                for i in range(end_index - start_index)
            ]
            # Compare each field for each event
            for field_name in self.data_map.constituents[constituent_name]:
                # Get ROOT data
                root_field = tree_data[constituent_name][field_name]

                # Compare event by event
                for i in range(len(valid_mask)):
                    event_valid = valid_mask[i]
                    n_valid = np.sum(event_valid)

                    if n_valid > 0:
                        # Get valid elements from ROOT (jagged array)
                        root_values = root_field[i][root_idxs[i]][:n_valid].astype(
                            np.float32
                        )
                        # Get valid elements from HDF5 (padded array)
                        h5_values = h5_const[field_name][i][event_valid]
                        if not np.allclose(
                            root_values,
                            h5_values,
                            equal_nan=True,
                            rtol=1e-6,
                            atol=1e-8,
                        ):
                            print(
                                f"  {constituent_name}.{field_name} at event {start_index + i}: Mismatch"
                            )
                            all_match = False

        if all_match:
            print("\nValidation successful! All samples match.")
        else:
            print("\nValidation failed! Some samples don't match.")

        return all_match


if __name__ == "__main__":
    args = get_parser()

    preparation_tool = PrepareSamples(args=args)
    preparation_tool.run()
    for _ in range(3):
        preparation_tool.validate(batch_size=1000)
