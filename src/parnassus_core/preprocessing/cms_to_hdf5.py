import argparse
import glob
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import uproot
from numpy.lib.recfunctions import unstructured_to_structured
from tqdm import tqdm

from parnassus_core.utils.data_utils import pdgid_class_dict

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


coord_var_list = ["pt", "eta", "phi", "mass"]
vertex_var_list = ["vx", "vy", "vz"]
additional_var_list = [
    "d0",
    "d0Error",
    "z0",
    "z0Error",
    "ip3d",
    "ip3dError",
]

var_dict = {
    "events": ["event"],
    "gens": coord_var_list + ["pdgId"] + vertex_var_list + ["label"],
    "pfcs": coord_var_list
    + ["pdgId"]
    + vertex_var_list
    + additional_var_list
    + ["label"],
}

prefix_dict = {
    "gens": "GenPart_",
    "pfcs": "PFCand_",
}


def get_parser():
    """Argument parser for CMS preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocess CMS files to HDF5 format")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input folder or file"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output filename", default="out.h5"
    )
    parser.add_argument(
        "-es", "--entry_start", type=int, help="Starting event", default=0
    )
    parser.add_argument(
        "-n", "--n_events", type=int, help="Number of events to process", default=100
    )
    parser.add_argument("-pt", "--pt_cut", type=float, help="Truth pt cut", default=0.0)
    parser.add_argument("-eta", "--eta_cut", type=float, help="Eta cut", default=2.7)
    parser.add_argument(
        "-m",
        "--max_particles",
        type=int,
        required=False,
        default=400,
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


class CMSPreprocessor:
    """Preprocess CMS data from ROOT files to HDF5 format."""

    def __init__(self, args):
        """Initialize the CMS preprocessor.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        """
        self.input_path = (
            Path(args.input)
            if not args.input.endswith(".root")
            else Path(args.input).parent
        )
        if args.input.endswith(".root"):
            self.input_files = [args.input]
        else:
            self.input_files = glob.glob(str(Path(args.input) / "*.root"))

        self.output = Path(args.output)
        self.entry_start = args.entry_start
        self.n_events = args.n_events
        self.pt_cut = args.pt_cut
        self.eta_cut = args.eta_cut
        self.max_particles = args.max_particles
        self.batch_size = args.batch_size

        # Data storage
        self.data = {
            name: {var: None for var in varlist} for name, varlist in var_dict.items()
        }

    def load_files(self):
        """Load data from ROOT files."""
        print("Loading data from ROOT files...")
        pbar_files = tqdm(self.input_files, leave=True, position=0)
        for file in pbar_files:
            pbar_files.set_description(f"Loading {Path(file).name}")
            with uproot.open(file) as f:
                pbar_trees = tqdm(var_dict.items(), leave=False, position=1)
                for name, varlist in pbar_trees:
                    pbar_trees.set_description(f"Loading tree {name}")
                    tree = f[name + "/Events"]
                    # Only load variables that exist in the tree
                    available_keys = tree.keys()
                    array_names = [
                        prefix_dict.get(name, "") + var
                        for var in varlist
                        if prefix_dict.get(name, "") + var in available_keys
                    ]
                    arrays = tree.arrays(
                        array_names,
                        library="np",
                        entry_stop=self.n_events + self.entry_start,
                        entry_start=self.entry_start,
                    )
                    for var in varlist:
                        array_name = prefix_dict.get(name, "") + var
                        if array_name not in available_keys:
                            continue  # Skip variables not present in file
                        val = arrays[array_name]
                        if self.data[name][var] is None:
                            self.data[name][var] = val
                        else:
                            self.data[name][var] = np.concatenate(
                                (self.data[name][var], val), axis=0
                            )

    def process_events(self):
        """Process events by applying cuts and computing derived quantities."""
        print("Processing events...")
        n_events = len(self.data["events"]["event"])
        self.data["pfcs"]["class"] = []
        self.data["gens"]["class"] = []
        n_gens = []
        n_pfcs = []

        for i in tqdm(range(n_events), desc="Applying cuts and filters"):
            # Compute class labels
            self.data["pfcs"]["class"].append(
                np.vectorize(pdgid_class_dict.get)(
                    self.data["pfcs"]["pdgId"][i]
                ).astype(int)
            )
            self.data["gens"]["class"].append(
                np.vectorize(pdgid_class_dict.get)(
                    self.data["gens"]["pdgId"][i]
                ).astype(int)
            )

            # Apply PFC mask
            pf_mask = (
                (self.data["pfcs"]["pt"][i] > 1)
                & (np.abs(self.data["pfcs"]["eta"][i]) < self.eta_cut)
                & (self.data["pfcs"]["class"][i] < 5)
            )

            # Apply truth mask
            neutrino_mask = (
                (np.abs(self.data["gens"]["pdgId"][i]) != 12)
                & (np.abs(self.data["gens"]["pdgId"][i]) != 14)
                & (np.abs(self.data["gens"]["pdgId"][i]) != 16)
            )
            tr_mask = (
                (self.data["gens"]["pt"][i] > self.pt_cut)
                & (np.abs(self.data["gens"]["eta"][i]) < self.eta_cut)
                & neutrino_mask
            )

            # Apply masks to all variables
            for var in (
                coord_var_list
                + ["pdgId", "class"]
                + vertex_var_list
                + additional_var_list
                + ["label"]
            ):
                if var in self.data["pfcs"] and self.data["pfcs"][var] is not None:
                    self.data["pfcs"][var][i] = self.data["pfcs"][var][i][pf_mask]
                if (
                    var not in additional_var_list
                    and var in self.data["gens"]
                    and self.data["gens"][var] is not None
                ):
                    self.data["gens"][var][i] = self.data["gens"][var][i][tr_mask]

            n_pfcs.append(len(self.data["pfcs"]["pt"][i]))
            n_gens.append(len(self.data["gens"]["pt"][i]))

        print(f"Max PFCs: {max(n_pfcs)}, Max Truths: {max(n_gens)}")
        print(f"Min PFCs: {min(n_pfcs)}, Min Truths: {min(n_gens)}")

        # Print variable ranges
        print("Truth variables: ")
        for var in self.data["gens"].keys():
            concat_var = np.concatenate(self.data["gens"][var])
            print(
                f"  {var}: Max={np.max(concat_var):.4f}, Min={np.min(concat_var):.4f}"
            )

        print("PFC variables: ")
        for var in self.data["pfcs"].keys():
            concat_var = np.concatenate(self.data["pfcs"][var])
            print(
                f"  {var}: Max={np.max(concat_var):.4f}, Min={np.min(concat_var):.4f}"
            )

        # Convert class arrays to object type for consistency
        self.data["pfcs"]["class"] = np.array(self.data["pfcs"]["class"], dtype=object)
        self.data["gens"]["class"] = np.array(self.data["gens"]["class"], dtype=object)

    def _convert_constituents(self, arrays, max_size, start_index=0, end_index=None):
        """Convert jagged arrays to padded structured arrays.

        Parameters
        ----------
        arrays : dict
            Dictionary of variable arrays
        max_size : int
            Maximum number of particles
        start_index : int
            Start index for slicing
        end_index : int or None
            End index for slicing

        Returns
        -------
        np.ndarray
            Structured array with padded data
        """
        dtypes = []
        output = []

        # Sort by pt
        sort_idxs = [
            np.argsort(-arrays["pt"][i])
            for i in range(
                start_index, end_index if end_index else arrays["pt"].shape[0]
            )
        ]

        for key in arrays.keys():
            arr = arrays[key]
            # Pad with NaNs to indicate invalid entries (use 0 for integer fields)
            is_integer_field = key in ["class", "label"]
            output.append(
                np.array(
                    [
                        np.pad(
                            arr[i][sort_idxs[i - start_index]][:max_size].astype(
                                np.float32
                            ),
                            (0, max_size - len(arr[i][:max_size])),
                            "constant",
                            constant_values=0 if is_integer_field else np.nan,
                        )
                        if arr[i].size > 0
                        else np.full(
                            max_size,
                            0 if is_integer_field else np.nan,
                            dtype=np.float32,
                        )
                        for i in range(
                            start_index, end_index if end_index else arr.shape[0]
                        )
                    ]
                )
            )
            dtypes.append((key, "f4"))

        # Create valid mask
        output.append(np.isfinite(output[0]) * 1)
        dtypes.append(("valid", "bool"))

        return unstructured_to_structured(
            np.stack(output, axis=-1), dtype=np.dtype(dtypes)
        )

    def _convert_events(self, events):
        """Convert event-level data to structured array.

        Parameters
        ----------
        events : dict
            Dictionary of event-level variables

        Returns
        -------
        np.ndarray
            Structured array with event data
        """
        event_dtypes = []
        events_output = []
        for key in events.keys():
            event_dtypes.append((key, "f4"))
            events_output.append(events[key].astype(np.float32))
        return unstructured_to_structured(
            np.stack(events_output, axis=-1), dtype=np.dtype(event_dtypes)
        )

    def _compute_event_features(self, arrays: np.ndarray) -> dict[str, np.ndarray]:
        """Compute derived event-level features.

        Parameters
        ----------
        arrays : np.ndarray
            Structured array of particle data

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of computed features
        """
        data_dict: dict[str, np.ndarray] = {}
        data_dict["ht"] = np.nansum(arrays["pt"], axis=1)
        data_dict["met_x"] = np.nansum(arrays["pt"] * np.cos(arrays["phi"]), axis=1)
        data_dict["met_y"] = np.nansum(arrays["pt"] * np.sin(arrays["phi"]), axis=1)
        return data_dict

    def run(self):
        """Run the preprocessing to convert CMS ROOT files to HDF5."""
        # Load and process data
        self.load_files()
        self.process_events()

        print("Converting and writing data to HDF5...")
        n_events = len(self.data["events"]["event"])
        batch_indices = [
            (i, min(i + self.batch_size, n_events))
            for i in range(0, n_events, self.batch_size)
        ]

        create_file = True
        displayed_writing_output = False
        pbar = tqdm(total=n_events)

        for start, end in batch_indices:
            h5_tree_data = {}

            # Build event-level dictionary
            event_data_dict = {"eventNumber": self.data["events"]["event"][start:end]}

            # Convert PFCs (pflow)
            pflow_arrays = {
                var: self.data["pfcs"][var][start:end]
                for var in self.data["pfcs"].keys()
                if self.data["pfcs"][var] is not None
            }
            h5_tree_data["pflow"] = self._convert_constituents(
                pflow_arrays, self.max_particles, start_index=0, end_index=end - start
            )

            # Compute event features from pflow
            for key, value in self._compute_event_features(
                h5_tree_data["pflow"]
            ).items():
                event_data_dict[f"pflow_{key}"] = value

            # Convert truth particles
            truth_arrays = {
                var: self.data["gens"][var][start:end]
                for var in self.data["gens"].keys()
                if self.data["gens"][var] is not None
            }
            h5_tree_data["truth"] = self._convert_constituents(
                truth_arrays, self.max_particles, start_index=0, end_index=end - start
            )

            # Compute event features from truth
            for key, value in self._compute_event_features(
                h5_tree_data["truth"]
            ).items():
                event_data_dict[f"truth_{key}"] = value

            # Convert global event variables
            h5_tree_data["events"] = self._convert_events(event_data_dict)

            pbar.update(end - start)

            if create_file:
                create_file = False
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
                # Append to existing dataset
                if displayed_writing_output:
                    pbar.write(f"Writing to output file: {self.output}")
                with h5py.File(self.output, "a") as out_file:
                    for key, value in h5_tree_data.items():
                        dataset = out_file[key]
                        dataset.resize((dataset.shape[0] + value.shape[0]), axis=0)
                        dataset[-value.shape[0] :] = value
                displayed_writing_output = False

        pbar.close()
        print(f"Successfully wrote {n_events} events to {self.output}")


def main():
    args = get_parser()
    preprocessor = CMSPreprocessor(args)
    preprocessor.run()


if __name__ == "__main__":
    main()
