import argparse
import glob
from collections import defaultdict
import json
from pathlib import Path

import awkward as ak
import h5py
import hdf5plugin
import numpy as np
import uproot
from tqdm import tqdm

from parnassus_core.utils.data_utils import pdgid_class_dict


PARTICLE_COLLECTIONS = {
    "gens": {
        "tree": "gens/Events",
        "prefix": "GenPart_",
        "output_name": "truth",
        "variables": ["pt", "eta", "phi", "mass", "pdgId", "vx", "vy", "vz", "label"],
        "optional": False,
    },
    "pfcs": {
        "tree": "pfcs/Events",
        "prefix": "PFCand_",
        "output_name": "pflow",
        "variables": [
            "pt",
            "eta",
            "phi",
            "mass",
            "pdgId",
            "vx",
            "vy",
            "vz",
            "d0",
            "d0Error",
            "z0",
            "z0Error",
            "ip3d",
            "ip3dError",
            "label",
        ],
        "optional": False,
    },
    "hadrons": {
        "tree": "hadrons/Events",
        "prefix": "HeavyHadron_",
        "output_name": "hadrons",
        "variables": ["pt", "eta", "phi", "mass", "pdgId", "vx", "vy", "vz", "label"],
        "optional": True,
    },
}

# Sim/Rec PV branches are intentionally omitted. cms_to_root writes them, but the
# current root_to_hdf5 converter excludes PV branches from its event schema.
EVENT_SCALARS = {
    "events": {
        "tree": "events/Events",
        "variables": {"event": "eventNumber"},
        "optional": False,
    },
    "pu": {
        "tree": "pu/Events",
        "variables": {"trueNumPU": None, "actualNumPU": None},
        "optional": True,
    },
}

DEFAULT_MAX_PARTICLES = defaultdict(lambda: 400, {"hadrons": 50})


LONG_VARIABLES = {"eventNumber"}
INT_VARIABLES = {"pdgId"}
SHORT_VARIABLES = {"class", "label", "charge"}


def get_parser():
    """Argument parser for direct CMS ROOT to HDF5 conversion."""
    parser = argparse.ArgumentParser(description="Preprocess CMS files to HDF5 format")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input folder or file"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output filename", default="out.h5"
    )
    parser.add_argument(
        "-es", "--entry_start", type=int, help="Starting event in each file", default=0
    )
    parser.add_argument(
        "-n",
        "--n_events",
        "--nevents",
        dest="n_events",
        type=int,
        help="Number of events to process; -1 processes all events",
        default=100,
    )
    parser.add_argument("-pt", "--pt_cut", type=float, help="Truth pt cut", default=0.0)
    parser.add_argument("-eta", "--eta_cut", type=float, help="Eta cut", default=2.7)
    parser.add_argument(
        "--pflow-pt-cut",
        type=float,
        default=1.0,
        help="PT cut for pflow particles (default: 1.0 GeV)",
    )
    parser.add_argument(
        "-m",
        "--max_particles",
        type=json.loads,
        required=False,
        help="Enter the maximum number of constituents to keep per type as a JSON string, "
        'e.g. \'{"pflow": 200, "truth": 100}\' (default: 400 for all types, except hadrons which defaults to 50).',
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=10_000,
        help="Number of events converted and written at a time",
    )
    parser.add_argument(
        "-nf",
        "--num_files",
        type=int,
        default=None,
        help="Maximum number of input files when input is a directory",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle events independently within each output batch",
    )
    parser.add_argument(
        "--cut-events",
        action="store_true",
        help=(
            "Remove events at or above a collection's particle limit, and events "
            "with no truth or pflow particles"
        ),
    )
    return parser.parse_args()


class CMSPreprocessor:
    """Convert POET CMS ROOT files directly to the Parnassus HDF5 layout."""

    def __init__(self, args):
        self.input = Path(args.input)
        self.input_files = self._get_input_files(
            self.input, getattr(args, "num_files", None)
        )
        self.output = Path(args.output)
        self.entry_start = args.entry_start
        self.n_events = args.n_events
        self.pt_cut = args.pt_cut
        self.eta_cut = args.eta_cut
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.cut_events = getattr(args, "cut_events", False)
        self.pflow_pt_cut = args.pflow_pt_cut

        if self.entry_start < 0:
            raise ValueError("entry_start must be non-negative")
        if self.n_events < -1:
            raise ValueError("n_events must be -1 or non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if args.max_particles is not None and (
            not isinstance(args.max_particles, dict)
            or not all(
                isinstance(name, str)
                and isinstance(limit, int)
                and not isinstance(limit, bool)
                and limit > 0
                for name, limit in args.max_particles.items()
            )
        ):
            raise ValueError(
                "max_particles must be a JSON object with positive integer values"
            )

        self.max_particles_dict = DEFAULT_MAX_PARTICLES.copy()
        if args.max_particles is not None:
            self.max_particles_dict |= args.max_particles

        self.rng = np.random.default_rng(seed=42)
        self.particle_data: dict[str, dict[str, ak.Array]] = {}
        self.event_data: dict[str, ak.Array] = {}
        self.removed_events = 0

    @staticmethod
    def _get_input_files(input_path: Path, num_files: int | None) -> list[Path]:
        if input_path.is_file():
            return [input_path]
        if not input_path.is_dir():
            raise ValueError(f"Input path does not exist: {input_path}")

        files = [Path(path) for path in glob.glob(str(input_path / "*.root"))]
        if num_files is not None and num_files > 0:
            files = files[:num_files]
        if not files:
            raise ValueError(f"No .root files found in directory: {input_path}")
        return files

    def _available_schema(self):
        """Find optional trees and branches present in every selected file."""
        particle_schema = {
            name: list(config["variables"])
            for name, config in PARTICLE_COLLECTIONS.items()
        }
        scalar_schema = {
            name: list(config["variables"]) for name, config in EVENT_SCALARS.items()
        }

        for filename in self.input_files:
            with uproot.open(filename) as root_file:
                for name, config in PARTICLE_COLLECTIONS.items():
                    if name not in particle_schema:
                        continue
                    if config["tree"] not in root_file:
                        if config["optional"]:
                            print(
                                f"Warning: optional tree {config['tree']} not found "
                                f"in file {filename}; omitting collection"
                            )
                            particle_schema.pop(name)
                            continue
                        raise ValueError(
                            f"Required tree {config['tree']} not found in file {filename}"
                        )

                    keys = set(root_file[config["tree"]].keys())
                    particle_schema[name] = [
                        var
                        for var in particle_schema[name]
                        if config["prefix"] + var in keys
                    ]

                for name, config in EVENT_SCALARS.items():
                    if name not in scalar_schema:
                        continue
                    if config["tree"] not in root_file:
                        if config["optional"]:
                            print(
                                f"Warning: optional tree {config['tree']} not found "
                                f"in file {filename}; omitting scalars"
                            )
                            scalar_schema.pop(name)
                            continue
                        raise ValueError(
                            f"Required tree {config['tree']} not found in file {filename}"
                        )

                    keys = set(root_file[config["tree"]].keys())
                    scalar_schema[name] = [
                        var for var in scalar_schema[name] if var in keys
                    ]

        return particle_schema, scalar_schema

    def load_files(self):
        """Load the requested event range from the CMS trees."""
        print(f"Loading data from {len(self.input_files)} ROOT file(s)...")
        particle_schema, scalar_schema = self._available_schema()

        particle_chunks = {
            name: {var: [] for var in variables}
            for name, variables in particle_schema.items()
        }
        scalar_chunks = {
            EVENT_SCALARS[name]["variables"][var] or var: []
            for name, variables in scalar_schema.items()
            for var in variables
        }

        remaining = None if self.n_events == -1 else self.n_events
        for filename in tqdm(self.input_files, desc="Loading ROOT files"):
            if remaining == 0:
                break

            with uproot.open(filename) as root_file:
                event_tree = root_file[EVENT_SCALARS["events"]["tree"]]
                available = max(0, event_tree.num_entries - self.entry_start)
                take = available if remaining is None else min(available, remaining)
                if take == 0:
                    continue

                entry_stop = self.entry_start + take
                for name, variables in particle_schema.items():
                    config = PARTICLE_COLLECTIONS[name]
                    tree = root_file[config["tree"]]
                    if tree.num_entries < entry_stop:
                        raise ValueError(
                            f"Tree {config['tree']} in {filename} has fewer entries "
                            "than events/Events"
                        )
                    branch_names = [config["prefix"] + var for var in variables]
                    arrays = tree.arrays(
                        branch_names,
                        library="ak",
                        entry_start=self.entry_start,
                        entry_stop=entry_stop,
                    )
                    for var, branch_name in zip(variables, branch_names):
                        particle_chunks[name][var].append(arrays[branch_name])

                for name, variables in scalar_schema.items():
                    config = EVENT_SCALARS[name]
                    tree = root_file[config["tree"]]
                    if tree.num_entries < entry_stop:
                        raise ValueError(
                            f"Tree {config['tree']} in {filename} has fewer entries "
                            "than events/Events"
                        )
                    arrays = tree.arrays(
                        variables,
                        library="ak",
                        entry_start=self.entry_start,
                        entry_stop=entry_stop,
                    )
                    for var in variables:
                        output_name = config["variables"][var] or var
                        scalar_chunks[output_name].append(arrays[var])

                if remaining is not None:
                    remaining -= take

        self.particle_data = {
            PARTICLE_COLLECTIONS[name]["output_name"]: {
                var: ak.concatenate(chunks, axis=0) for var, chunks in variables.items()
            }
            for name, variables in particle_chunks.items()
            if variables and all(chunks for chunks in variables.values())
        }
        self.event_data = {
            name: ak.concatenate(chunks, axis=0)
            for name, chunks in scalar_chunks.items()
            if chunks
        }

        if "eventNumber" not in self.event_data:
            raise ValueError("No eventNumber data found")

    @staticmethod
    def _derive_class(pdg_id: ak.Array) -> ak.Array:
        flat_pdg_id = ak.to_numpy(ak.flatten(pdg_id))
        flat_class = np.vectorize(pdgid_class_dict.get)(flat_pdg_id).astype(np.int32)
        return ak.unflatten(flat_class, ak.num(pdg_id))

    def process_events(self):
        """Apply the same filters and derived fields as cms_to_root.py."""
        print("Processing particle collections...")

        for collection_name in ("pflow", "truth"):
            if collection_name not in self.particle_data:
                continue
            collection = self.particle_data[collection_name]
            pdg_id = collection["pdgId"]
            particle_class = self._derive_class(pdg_id)
            charge = ak.values_astype(np.sign(pdg_id), np.int32)
            collection["charge"] = ak.where(particle_class > 2, 0, charge)
            collection["class"] = particle_class

            if collection_name == "pflow":
                mask = (
                    (collection["pt"] > self.pflow_pt_cut)
                    & (np.abs(collection["eta"]) < self.eta_cut)
                    & (collection["class"] < 5)
                )
            else:
                mask = (
                    (collection["pt"] > self.pt_cut)
                    & (np.abs(collection["eta"]) < self.eta_cut)
                    & (np.abs(pdg_id) != 12)
                    & (np.abs(pdg_id) != 14)
                    & (np.abs(pdg_id) != 16)
                )

            for var in collection:
                collection[var] = collection[var][mask]

        for name, collection in self.particle_data.items():
            counts = ak.to_numpy(ak.num(collection["pt"]))
            if len(counts):
                print(f"{name} - Max: {counts.max()}, Min: {counts.min()}")

    @staticmethod
    def _field_dtype(name: str):
        if name in LONG_VARIABLES:
            return np.int64
        if name in INT_VARIABLES:
            return np.int32
        if name in SHORT_VARIABLES:
            return np.int16
        return np.float32

    def _convert_constituents(
        self,
        arrays: dict[str, ak.Array],
        max_size: int,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort, truncate, pad, and type one particle collection."""
        end_index = len(arrays["pt"]) if end_index is None else end_index
        n_events = end_index - start_index
        field_dtypes = [(key, self._field_dtype(key)) for key in arrays]
        field_dtypes.append(("valid", np.bool_))
        output = np.empty((n_events, max_size), dtype=np.dtype(field_dtypes))

        for key in arrays:
            pad_value = (
                0 if key in LONG_VARIABLES | INT_VARIABLES | SHORT_VARIABLES else np.nan
            )
            output[key].fill(pad_value)
        output["valid"].fill(False)

        original_length = np.empty(n_events, dtype=np.int32)
        for output_index, event_index in enumerate(range(start_index, end_index)):
            sort_source = np.asarray(arrays["pt"][event_index])
            event_length = len(sort_source)
            original_length[output_index] = event_length
            n_keep = min(event_length, max_size)
            if n_keep == 0:
                continue

            if event_length <= max_size:
                take_idx = np.argsort(-sort_source)
            else:
                take_idx = np.argpartition(-sort_source, max_size - 1)[:max_size]
                take_idx = take_idx[np.argsort(-sort_source[take_idx])]

            for key, values in arrays.items():
                event_values = np.asarray(values[event_index])
                output[key][output_index, :n_keep] = event_values[take_idx[:n_keep]]
            output["valid"][output_index, :n_keep] = True

        return output, original_length

    @staticmethod
    def _compute_event_features(arrays: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "ht": np.nansum(arrays["pt"], axis=1),
            "met_x": np.nansum(arrays["pt"] * np.cos(arrays["phi"]), axis=1),
            "met_y": np.nansum(arrays["pt"] * np.sin(arrays["phi"]), axis=1),
            "npart": np.sum(arrays["valid"], axis=1).astype(np.int32),
        }

    def _convert_events(self, events: dict[str, np.ndarray]) -> np.ndarray:
        output = np.empty(
            len(next(iter(events.values()))),
            dtype=np.dtype([(key, self._field_dtype(key)) for key in events]),
        )
        for key, values in events.items():
            output[key] = np.asarray(values, dtype=self._field_dtype(key))
        return output

    def _convert_batch(self, start: int, end: int) -> dict[str, np.ndarray]:
        output = {}
        event_fields = {
            key: ak.to_numpy(values[start:end])
            for key, values in self.event_data.items()
        }

        for collection_name, arrays in self.particle_data.items():
            converted, original_length = self._convert_constituents(
                arrays,
                self.max_particles_dict[collection_name],
                start_index=start,
                end_index=end,
            )
            output[collection_name] = converted
            event_fields[f"{collection_name}_npart_orig"] = original_length
            for feature, values in self._compute_event_features(converted).items():
                event_fields[f"{collection_name}_{feature}"] = values

        output["events"] = self._convert_events(event_fields)
        return output

    def _filter_and_shuffle(self, batch: dict[str, np.ndarray]):
        if self.cut_events:
            valid = np.ones(len(batch["events"]), dtype=bool)
            for collection_name in self.particle_data:
                original_length = batch["events"][f"{collection_name}_npart_orig"]
                valid &= original_length < self.max_particles_dict[collection_name]
                if collection_name in {"truth", "pflow"}:
                    valid &= original_length > 0

            self.removed_events += int(np.sum(~valid))
            for key in batch:
                batch[key] = batch[key][valid]

        if self.shuffle:
            indices = np.arange(len(batch["events"]))
            self.rng.shuffle(indices)
            for key in batch:
                batch[key] = batch[key][indices]

    @staticmethod
    def _chunk_shape(key: str, value: np.ndarray):
        if key == "events":
            return (max(1, min(1024, len(value))),)
        return (max(1, min(64, len(value))), value.shape[1])

    def run(self):
        """Run the direct conversion."""
        self.load_files()
        self.process_events()

        n_events = len(self.event_data["eventNumber"])
        print(f"Converting and writing {n_events} events to {self.output}...")
        datasets: dict[str, h5py.Dataset] = {}

        with h5py.File(self.output, "w") as output_file:
            for start in tqdm(range(0, n_events, self.batch_size), desc="Writing"):
                end = min(start + self.batch_size, n_events)
                batch = self._convert_batch(start, end)
                self._filter_and_shuffle(batch)

                if not datasets:
                    for key, value in batch.items():
                        maxshape = (
                            (None,) if key == "events" else (None, value.shape[1])
                        )
                        datasets[key] = output_file.create_dataset(
                            key,
                            data=value,
                            maxshape=maxshape,
                            chunks=self._chunk_shape(key, value),
                            compression=hdf5plugin.LZ4(),
                        )
                else:
                    for key, value in batch.items():
                        dataset = datasets[key]
                        old_size = dataset.shape[0]
                        dataset.resize(old_size + len(value), axis=0)
                        dataset[old_size:] = value

            if not datasets:
                empty_batch = self._convert_batch(0, 0)
                for key, value in empty_batch.items():
                    maxshape = (None,) if key == "events" else (None, value.shape[1])
                    output_file.create_dataset(
                        key,
                        data=value,
                        maxshape=maxshape,
                        chunks=self._chunk_shape(key, value),
                        compression=hdf5plugin.LZ4(),
                    )

        written_events = n_events - self.removed_events
        print(f"Successfully wrote {written_events} events to {self.output}")
        if self.cut_events:
            print(f"Removed {self.removed_events} events due to particle limits")


def main():
    args = get_parser()
    CMSPreprocessor(args).run()


if __name__ == "__main__":
    main()
