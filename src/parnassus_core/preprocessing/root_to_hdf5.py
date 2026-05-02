import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, cast, final

import h5py
import hdf5plugin
import numpy as np
import uproot

from tqdm import tqdm

## Example ROOT file structure:
#  Branches:
#    ntruth           ->  int32_t
#    truth_pt         ->  float[]
#    truth_eta        ->  float[]
#    truth_phi        ->  float[]
#    truth_class      ->  int32_t[]
#    npflow           ->  int32_t
#    pflow_pt         ->  float[]
#    pflow_eta        ->  float[]
#    pflow_phi        ->  float[]
#    pflow_class      ->  int32_t[]
#    nhadrons         ->  int32_t
#    hadrons_pt       ->  float[]
#    hadrons_eta      ->  float[]
#    hadrons_phi      ->  float[]
#    hadrons_mass     ->  float[]
#    hadrons_pdgId    ->  int32_t[]
#    hadrons_vx       ->  float[]
#    hadrons_vy       ->  float[]
#    hadrons_vz       ->  float[]
#    hadrons_label    ->  int32_t[]
#    eventNumber      ->  uint64_t
#    trueNumPU        ->  float
#    actualNumPU      ->  float

MAX_PARTICLES_DICT = defaultdict(lambda: 400)
MAX_PARTICLES_DICT.update({"hadrons": 50})

INT_VARIABLES = {"pdgId"}
SHORT_VARIABLES = {"class", "label", "charge"}


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
        help="Enter the name of the input file or directory containing .root files.",
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

    parser.add_argument(
        "-nf",
        "--num_files",
        type=int,
        required=False,
        default=None,
        help="Number of files to load from directory (default: all files).",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the events before writing to HDF5 (default: False).",
    )

    parser.add_argument(
        "--cut-events",
        action="store_true",
        help=(
            "Whether to cut events that exceed max_particles (default: False). "
            "If enabled, only events with npart_orig < max_particles for all constituents will be kept."
        ),
    )

    return parser.parse_args()


@dataclass
class TreeMap:
    # Define the structure for mapping tree variables
    tree_name: str
    global_vars: list[str]  # e.g., ['eventNumber', ...]
    constituents: dict[str, list[str]]  # e.g., {'tracks': [...], 'cells': [...]}

    @classmethod
    def create(cls, tree_name: str, tree) -> "TreeMap":
        """Create TreeMap by analyzing branch dtypes.

        Scalar branches -> global event variables
        Array branches -> constituent variables (grouped by prefix)
        """
        constituents = {}
        global_vars = []

        for key in tree.keys():
            branch = tree[key]
            # Check if branch is an array type (jagged or variable-length)
            is_array = hasattr(branch.interpretation, "inner") or str(
                branch.typename
            ).endswith("[]")

            if is_array:
                # Array branch: constituent variable
                # Skip count variables (e.g., 'npflow', 'ntruth')
                if key.startswith("n") and len(key) > 1 and "_" not in key:
                    continue

                # Extract constituent name and variable name
                if "_" in key:
                    constituent_name, var_name = key.split("_", 1)
                    if constituent_name not in constituents:
                        constituents[constituent_name] = []
                    constituents[constituent_name].append(var_name)
            else:
                # Scalar branch: global event variable
                # Skip count variables
                if key.startswith("n") and len(key) > 1 and "_" not in key:
                    continue
                # Skip PV variables for now...
                if "PV" in key:
                    continue
                global_vars.append(key)

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

        # Get list of input files (single file or all .root files in directory)
        self.input_files = self._get_input_files(self.input, args.num_files)

        if args.output is None:
            if self.input.is_dir():
                self.output = self.input / "output.h5"
            else:
                self.output = self.input.with_suffix(".h5")
        else:
            self.output = Path(args.output)

        self.n_events = args.nevents
        self.max_particles_dict = (
            args.max_particles if args.max_particles is not None else MAX_PARTICLES_DICT
        )
        self.batch_size = args.batch_size
        self.cut_events = args.cut_events
        self.shuffle = args.shuffle

        self.data_map, total_events = self.get_metadata(self.input_files)
        if self.n_events is None or self.n_events > total_events:
            self.n_events = total_events

        self.removed_events = 0

        self.rng = np.random.default_rng(seed=42)  # for reproducibility

    def _get_input_files(
        self, input_path: Path, num_files: int | None = None
    ) -> list[Path]:
        """Get list of input ROOT files.

        Parameters
        ----------
        input_path : Path
            Path to a single file or directory containing .root files.
        num_files : int | None
            Maximum number of files to load from directory. If None, loads all files.
            Only applies when input_path is a directory.

        Returns
        -------
        list[Path]
            List of ROOT file paths.
        """
        if input_path.is_file():
            return [input_path]
        elif input_path.is_dir():
            root_files = sorted(input_path.glob("*.root"))
            if not root_files:
                raise ValueError(f"No .root files found in directory: {input_path}")

            # Limit number of files if specified
            if num_files is not None and num_files > 0:
                root_files = root_files[:num_files]
                print(
                    f"Loading {len(root_files)} of {len(sorted(input_path.glob('*.root')))} .root files from {input_path}"
                )
            else:
                print(f"Found {len(root_files)} .root files in {input_path}")

            return root_files
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def get_metadata(self, filenames: list[Path]) -> tuple[TreeMap, int]:
        """
        Get the data map from the ROOT file(s).

        Parameters
        ----------
        filenames : list[Path]
            List of paths to ROOT files.
        Returns
        -------
        tuple[TreeMap, int]
            The TreeMap object and the total number of entries across all files.
        """
        # Get metadata from the first file
        with uproot.open(filenames[0]) as rootfile:  # type: ignore
            tree = rootfile.keys()[0]  # get first tree
            tree_obj = rootfile[tree]
            tree_map = TreeMap.create(tree, tree_obj)

        # Count total events across all files
        total_events = 0
        for filename in filenames:
            with uproot.open(filename) as rootfile:  # type: ignore
                tree_obj = rootfile[tree_map.tree_name]
                total_events += tree_obj.num_entries

        print(f"Total events across {len(filenames)} file(s): {total_events}")
        return tree_map, total_events

    def _load_events(
        self, filenames: list[Path], start: int, end: int
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Load events from ROOT file(s).

        For a single file, loads events from start to end index.
        For multiple files, loads ALL events from ALL files (ignores start/end).

        Parameters
        ----------
        filenames : list[Path]
            List of ROOT file paths.
        start : int
            Starting event index (only used for single file).
        end : int
            Ending event index (only used for single file).

        Returns
        -------
        dict
            Dictionary containing the loaded and concatenated event data.
        """
        tree_name = self.data_map.tree_name
        tree_data = {}

        branches = list(self.data_map.global_vars)
        for constituent_name, var_names in self.data_map.constituents.items():
            branches.extend(f"{constituent_name}_{var_name}" for var_name in var_names)

        # For multiple files, use one concatenate call to avoid repeated
        # np.concatenate copies inside a per-file loop.
        if len(filenames) > 1:
            print(f"Loading all events from {len(filenames)} files...")
            file_specs = [f"{filename}:{tree_name}" for filename in filenames]
            arrays = uproot.concatenate(file_specs, expressions=branches, library="np")

            for global_var in self.data_map.global_vars:
                tree_data[global_var] = arrays[global_var]

            for constituent_name, var_names in self.data_map.constituents.items():
                tree_data[constituent_name] = {
                    var_name: arrays[f"{constituent_name}_{var_name}"]
                    for var_name in var_names
                }
        else:
            # For a single file, use start/end indices
            with uproot.open(filenames[0]) as rootfile:  # type: ignore
                tree = rootfile[tree_name]
                arrays = tree.arrays(
                    expressions=branches,
                    entry_start=start,
                    entry_stop=end,
                    library="np",
                )

                # Initialize global variables
                for global_var in self.data_map.global_vars:
                    tree_data[global_var] = arrays[global_var]

                # Initialize constituent arrays
                for constituent_name, var_names in self.data_map.constituents.items():
                    tree_data[constituent_name] = {
                        var_name: arrays[f"{constituent_name}_{var_name}"]
                        for var_name in var_names
                    }

        return tree_data

    def _convert_events(self, events):
        event_dtype = np.dtype([(key, "f4") for key in events])
        n_events = len(next(iter(events.values()))) if events else 0
        out = np.empty(n_events, dtype=event_dtype)
        for key, values in events.items():
            out[key] = np.asarray(values, dtype=np.float32)
        return out

    def _convert_constituents(self, arrays, max_size, start_index=0, end_index=None):
        slice_obj = (
            slice(start_index, end_index)
            if end_index is not None
            else slice(start_index, None)
        )

        # Sort by pt when available, otherwise by energy.
        sort_source_name = "pt" if "pt" in arrays else "e"
        sort_source = arrays[sort_source_name][slice_obj]
        n_events = len(sort_source)

        # Use partial sort for long events: pick top-k with argpartition,
        # then fully sort only those k entries.
        sort_idxs = []
        for i in range(n_events):
            event_source = sort_source[i]
            event_len = len(event_source)
            if event_len <= max_size:
                sort_idxs.append(np.argsort(-event_source))
            else:
                topk = np.argpartition(-event_source, max_size - 1)[:max_size]
                topk = topk[np.argsort(-event_source[topk])]
                sort_idxs.append(topk)

        original_length = np.fromiter(
            (len(ev) for ev in sort_source), dtype=np.int32, count=n_events
        )
        clipped_length = np.minimum(original_length, max_size)
        arr_subsets = {key: arr[slice_obj] for key, arr in arrays.items()}

        field_dtypes = []
        field_np_dtypes = {}
        for key in arrays.keys():
            if key in INT_VARIABLES:
                field_dtypes.append((key, "i4"))
                field_np_dtypes[key] = np.int32
            elif key in SHORT_VARIABLES:
                field_dtypes.append((key, "i2"))
                field_np_dtypes[key] = np.int16
            else:
                field_dtypes.append((key, "f4"))
                field_np_dtypes[key] = np.float32
        field_dtypes.append(("valid", "bool"))

        out = np.empty((n_events, max_size), dtype=np.dtype(field_dtypes))

        # Initialize pads in-place in the structured output.
        for key in arrays.keys():
            if key in INT_VARIABLES or key in SHORT_VARIABLES:
                out[key].fill(0)
            else:
                out[key].fill(np.nan)
        out["valid"].fill(False)

        # Fill one event at a time to avoid per-field intermediate arrays.
        for i in range(n_events):
            n_keep = clipped_length[i]
            if n_keep <= 0:
                continue
            take_idx = sort_idxs[i][:n_keep]

            for key, arr_subset in arr_subsets.items():
                event_arr = arr_subset[i]
                if event_arr.size == 0:
                    continue
                out[key][i, :n_keep] = event_arr[take_idx].astype(
                    field_np_dtypes[key], copy=False
                )

            out["valid"][i, :n_keep] = True

        return out, original_length

    def _compute_event_features(self, arrays: np.ndarray) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        # Only compute features if required fields exist
        if "pt" in arrays.dtype.names:
            data_dict["ht"] = np.nansum(arrays["pt"], axis=1)
        if "pt" in arrays.dtype.names and "phi" in arrays.dtype.names:
            data_dict["met_x"] = np.nansum(arrays["pt"] * np.cos(arrays["phi"]), axis=1)
            data_dict["met_y"] = np.nansum(arrays["pt"] * np.sin(arrays["phi"]), axis=1)
        if "valid" in arrays.dtype.names:
            data_dict["npart"] = np.sum(arrays["valid"], axis=1).astype(np.int32)
        return data_dict

    def _iter_event_chunks(
        self,
    ) -> "Generator[dict[str, np.ndarray | dict[str, np.ndarray]], None, None]":
        """Yield event chunks directly from ROOT files.

        This avoids materializing all events in memory before conversion.
        """
        tree_name = self.data_map.tree_name

        branches = list(self.data_map.global_vars)
        for constituent_name, var_names in self.data_map.constituents.items():
            branches.extend(f"{constituent_name}_{var_name}" for var_name in var_names)

        file_specs = [f"{filename}:{tree_name}" for filename in self.input_files]
        for arrays in uproot.iterate(
            file_specs,
            expressions=branches,
            library="np",
            step_size=self.batch_size,
        ):
            tree_data: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
            for global_var in self.data_map.global_vars:
                tree_data[global_var] = arrays[global_var]  # type: ignore
            for constituent_name, var_names in self.data_map.constituents.items():
                tree_data[constituent_name] = {
                    var_name: arrays[f"{constituent_name}_{var_name}"]  # type: ignore
                    for var_name in var_names
                }
            yield tree_data

    def run(self):
        """Run the preprocessing to convert ROOT to HDF5."""
        print("Converting and writing data...")
        displayed_writing_output = False
        pbar = tqdm(total=self.n_events)
        processed_events = 0
        write_offset = 0
        preallocate_output = not self.cut_events

        datasets: dict[str, h5py.Dataset] = {}
        with h5py.File(self.output, "w") as out_file:
            for tree_data in self._iter_event_chunks():
                if processed_events >= self.n_events:
                    break

                if self.data_map.global_vars:
                    chunk_size = len(
                        cast(np.ndarray, tree_data[self.data_map.global_vars[0]])
                    )
                else:
                    first_constituent = next(iter(self.data_map.constituents))
                    first_var = self.data_map.constituents[first_constituent][0]
                    chunk_size = len(
                        cast(dict[str, np.ndarray], tree_data[first_constituent])[
                            first_var
                        ]
                    )

                keep = min(chunk_size, self.n_events - processed_events)
                if keep <= 0:
                    break

                h5_tree_data = {}
                event_data_dict = {
                    k: cast(np.ndarray, tree_data[k])[:keep]
                    for k in self.data_map.global_vars
                }

                # convert constituents
                for constituent_name in self.data_map.constituents:
                    h5_tree_data[constituent_name], original_length = (
                        self._convert_constituents(
                            cast(dict[str, np.ndarray], tree_data[constituent_name]),
                            self.max_particles_dict[constituent_name],
                            start_index=0,
                            end_index=keep,
                        )
                    )
                    event_data_dict[f"{constituent_name}_npart_orig"] = (
                        original_length.astype(np.int32)
                    )
                    for key, value in self._compute_event_features(
                        h5_tree_data[constituent_name]
                    ).items():
                        event_data_dict[f"{constituent_name}_{key}"] = value
                # convert global event variables
                h5_tree_data["events"] = self._convert_events(event_data_dict)
                pbar.update(keep)
                processed_events += keep

                if self.cut_events:
                    valid_mask = np.ones(keep, dtype=bool)
                    for consituent_name in self.data_map.constituents:
                        valid_mask &= (
                            h5_tree_data["events"][f"{consituent_name}_npart_orig"]
                            < self.max_particles_dict[consituent_name]
                        )
                        if consituent_name in {"pflow", "truth"}:
                            valid_mask &= (
                                h5_tree_data["events"][f"{consituent_name}_npart_orig"]
                                > 0
                            )
                    h5_tree_data["events"] = h5_tree_data["events"][valid_mask]
                    for constituent_name in self.data_map.constituents:
                        h5_tree_data[constituent_name] = h5_tree_data[constituent_name][
                            valid_mask
                        ]
                    self.removed_events += np.sum(~valid_mask)

                if self.shuffle:
                    idx = np.arange(h5_tree_data["events"].shape[0])
                    self.rng.shuffle(idx)
                    for key in h5_tree_data:
                        h5_tree_data[key] = h5_tree_data[key][idx]

                if not datasets:
                    pbar.write(f"Creating output file: {self.output}")
                    displayed_writing_output = True
                    for key, value in h5_tree_data.items():
                        if key == "events":
                            chunk_shape = (1024,)
                            if preallocate_output:
                                shape = (self.n_events,)
                                max_shape = (self.n_events,)
                            else:
                                shape = value.shape
                                max_shape = (None,)
                        else:
                            width = self.max_particles_dict[key]
                            # For constituents, we set chunk shape to (64, width) to optimize for reading batches of events
                            chunk_shape = (64, width)
                            if preallocate_output:
                                shape = (self.n_events, width)
                                max_shape = (self.n_events, width)
                            else:
                                shape = value.shape
                                max_shape = (None, width)

                        if preallocate_output:
                            datasets[key] = out_file.create_dataset(
                                key,
                                shape=shape,
                                dtype=value.dtype,
                                chunks=chunk_shape,
                                maxshape=max_shape,
                                compression=hdf5plugin.LZ4(),  # pyright: ignore[reportPrivateImportUsage]
                            )
                        else:
                            datasets[key] = out_file.create_dataset(
                                key,
                                data=value,
                                chunks=chunk_shape,
                                maxshape=max_shape,
                                compression=hdf5plugin.LZ4(),  # pyright: ignore[reportPrivateImportUsage]
                            )

                    if preallocate_output:
                        batch_written = h5_tree_data["events"].shape[0]
                        for key, value in h5_tree_data.items():
                            datasets[key][
                                write_offset : write_offset + batch_written
                            ] = value
                        write_offset += batch_written
                else:
                    # appending to existing dataset
                    if displayed_writing_output:
                        pbar.write(f"Writing to output file: {self.output}")
                    if preallocate_output:
                        batch_written = h5_tree_data["events"].shape[0]
                        for key, value in h5_tree_data.items():
                            datasets[key][
                                write_offset : write_offset + batch_written
                            ] = value
                        write_offset += batch_written
                    else:
                        for key, value in h5_tree_data.items():
                            dataset = datasets[key]
                            dataset.resize((dataset.shape[0] + value.shape[0]), axis=0)
                            dataset[-value.shape[0] :] = value
                    displayed_writing_output = False
        pbar.close()

        print(
            f"Finished writing {self.n_events - self.removed_events} events to {self.output}"
        )
        if self.cut_events:
            print(
                f"Removed {self.removed_events} events that exceeded max_particles limit."
            )

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
        tree_data = self._load_events(self.input_files, start_index, end_index)

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

    if (
        len(preparation_tool.input_files) == 1
        and not preparation_tool.shuffle
        and not preparation_tool.cut_events
    ):
        preparation_tool.validate(batch_size=1000)
