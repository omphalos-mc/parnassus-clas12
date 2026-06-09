import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import hdf5plugin
import numpy as np
from numpy.lib.recfunctions import append_fields
from tqdm import tqdm


def get_parser():
    """
    Argument parser for HDF5 merge and shuffle script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(
        description="Merge and shuffle multiple HDF5 files."
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        required=True,
        help="Input HDF5 files. Can be: space-separated list of files, or glob pattern (e.g., 'data/*.h5').",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Output HDF5 file path. With --events-per-file, numbered files "
            "such as output_00000.h5 are created."
        ),
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        required=False,
        default=10_000,
        help="Batch size for writing events to output file.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for shuffling. Default is 42.",
    )

    parser.add_argument(
        "-n",
        "--nevents",
        type=int,
        required=False,
        default=None,
        help="Maximum number of events to process. If not provided, all events are used.",
    )

    parser.add_argument(
        "--equal-events",
        action="store_true",
        help="Take equal number of events from each file (uses min events across all files).",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help=(
            "Disable shuffling and concatenate inputs sequentially: all events "
            "from the first file, then the second file, and so on."
        ),
    )

    parser.add_argument(
        "--events-per-file",
        type=int,
        default=None,
        help=(
            "Approximate number of events per output file. Files rotate after "
            "a complete write batch, so they may contain slightly more events."
        ),
    )

    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help=(
            "Skip checking for overlapping event numbers between input files. "
            "This can speed up processing if you are sure there are no overlaps."
        ),
    )

    return parser.parse_args()


def discover_input_files(input_list: list[str]) -> list[Path]:
    """
    Discover input files from list of strings (glob patterns or file paths).

    Parameters
    ----------
    input_list : list[str]
        List of file paths or glob patterns.

    Returns
    -------
    list[Path]
        List of absolute paths to HDF5 files.
    """
    file_paths = []

    # Process each input string (could be glob pattern or direct path)
    for input_str in input_list:
        # Check if input contains glob wildcards
        if any(char in input_str for char in ["*", "?", "["]):
            # Use glob pattern
            files = glob.glob(input_str)
            if not files:
                raise ValueError(f"No files found matching pattern: {input_str}")
            file_paths.extend([Path(f).resolve() for f in files])
        else:
            # Direct file path
            file_paths.append(Path(input_str).resolve())

    # Validate all files exist and are HDF5
    for fpath in file_paths:
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
        if not fpath.suffix == ".h5":
            raise ValueError(f"Not an HDF5 file: {fpath}")

    print(f"Found {len(file_paths)} input file(s)")
    return sorted(file_paths)


class MergeShuffleHDF5:
    """
    Merge and shuffle multiple HDF5 files into one or more output files.

    This class reads multiple HDF5 files (outputs from root_to_hdf5.py),
    shuffles all events globally, and writes them to new HDF5 files
    with source file tracking.
    """

    def __init__(self, args) -> None:
        """
        Initialize merger with input files and parameters.

        Parameters
        ----------
        args : parse_args
            Command line arguments
        """
        self.output = Path(args.output)
        self.batch_size = args.batch_size
        if self.batch_size <= 0:
            raise ValueError("--batch-size must be greater than zero")
        self.seed = args.seed
        self.n_events_limit = args.nevents
        self.equal_events = args.equal_events
        self.no_shuffle = args.no_shuffle
        self.events_per_output_file = args.events_per_file
        if self.events_per_output_file is not None and self.events_per_output_file <= 0:
            raise ValueError("--events-per-file must be greater than zero")

        # Discover input files
        self.file_handles: list[h5py.File] = []

        self.input_files = discover_input_files(args.input)
        # Open all input files and store handles
        print("Opening input files...")
        for fpath in self.input_files:
            self.file_handles.append(h5py.File(fpath, "r"))

        # Extract dataset structure from first file
        first_file = self.file_handles[0]
        self.dataset_keys = list(first_file.keys())
        print(f"Datasets found: {self.dataset_keys}")

        # Separate events from constituents
        self.constituent_names = [k for k in self.dataset_keys if k != "events"]

        # Count events per file
        self.events_per_file = []
        for f in self.file_handles:
            n_events = cast(h5py.Dataset, f["events"]).shape[0]
            self.events_per_file.append(n_events)

        print(
            f"Events per file: {self.events_per_file} (total: {sum(self.events_per_file)})"
        )

        # Check for overlapping event numbers
        if not args.skip_overlap_check:
            self._check_event_number_overlaps()

        # Apply equal events constraint if specified
        if self.equal_events:
            min_events = min(self.events_per_file)
            self.events_per_file_limited = [min_events] * len(self.events_per_file)
            print(f"Equal events mode: using {min_events} events from each file")
            print(f"Total events to process: {min_events * len(self.input_files)}")
        else:
            self.events_per_file_limited = self.events_per_file.copy()

        # Determine total events to process
        self.total_events = sum(self.events_per_file_limited)
        if self.n_events_limit is not None and self.n_events_limit < self.total_events:
            self.total_events = self.n_events_limit
            print(f"Limiting to {self.total_events} events")

        self.rng = np.random.default_rng(self.seed)
        if self.no_shuffle:
            print(
                f"Sequential mode: reading {self.batch_size} events at a time "
                "from each input file in order"
            )
        else:
            # Calculate file-specific batch sizes for better mixing
            self.batch_sizes_per_file = self._calculate_batch_sizes_per_file()
            total_to_use = sum(self.events_per_file_limited)
            print("Batch sizes per file:")
            for i, (batch_size, n_events_limited) in enumerate(
                zip(self.batch_sizes_per_file, self.events_per_file_limited)
            ):
                proportion = n_events_limited / total_to_use
                actual_events = self.events_per_file[i]
                if self.equal_events:
                    print(
                        f"  File {i} ({self.input_files[i].name}): {batch_size:6d} ({proportion:6.2%} of events, using {n_events_limited}/{actual_events})"
                    )
                else:
                    print(
                        f"  File {i} ({self.input_files[i].name}): {batch_size:6d} ({proportion:6.2%} of events)"
                    )
            print(f"  Total batch size: {sum(self.batch_sizes_per_file)}")

            # Create batch descriptors for mixed-file reading
            print(f"Creating batch descriptors (seed={self.seed})...")
            self.batch_descriptors = self._create_batch_descriptors()

    def _check_event_number_overlaps(self) -> None:
        """
        Check for overlapping event numbers between files and print warnings.
        """
        print("Checking for event number overlaps...")

        # Load all event numbers once
        event_numbers = []
        for i, f in enumerate(self.file_handles):
            events_dataset = cast(h5py.Dataset, f["events"])
            ev_nums = events_dataset["eventNumber"][:]
            event_numbers.append(set(ev_nums))

        # Check pairwise intersections
        overlaps_found = False
        for i in range(len(event_numbers)):
            for j in range(i + 1, len(event_numbers)):
                intersection = event_numbers[i] & event_numbers[j]
                if intersection:
                    overlaps_found = True
                    print(
                        f"WARNING: Files {self.input_files[i].name} and "
                        f"{self.input_files[j].name} have {len(intersection)} "
                        f"overlapping event numbers"
                    )

        if not overlaps_found:
            print("No event number overlaps detected")

    def _calculate_batch_sizes_per_file(self) -> list[int]:
        """
        Calculate file-specific batch sizes proportional to each file's share
        of total events. The sum of all batch sizes equals the user-specified
        batch_size.

        For example, if file A has twice as many events as file B, its batch
        size will be twice as large. When batches from all files are combined,
        the total batch size matches the specified batch_size.

        Returns
        -------
        list[int]
            Batch size for each input file
        """
        total_original_events = sum(self.events_per_file_limited)

        batch_sizes = []
        allocated_total = 0

        for i, n_events in enumerate(self.events_per_file_limited):
            # Calculate proportional batch size
            file_proportion = n_events / total_original_events

            if i == len(self.events_per_file) - 1:
                # Last file: allocate remaining to ensure exact sum
                file_batch_size = self.batch_size - allocated_total
            else:
                # Proportional allocation
                file_batch_size = int(self.batch_size * file_proportion)
                allocated_total += file_batch_size

            # Ensure batch size is at least 1 and at most the file's total events
            file_batch_size = max(1, min(file_batch_size, n_events))

            batch_sizes.append(file_batch_size)

        return batch_sizes

    def _create_batch_descriptors(self) -> list[list[dict]]:
        """
        Create batch descriptors organized per-file.
        Each file gets a list of batch descriptors for sequential reading.

        Returns
        -------
        list[list[dict]]
            List of batch descriptor lists, one per file.
            Each descriptor has keys: file_idx, start_idx, end_idx, source_file_idx
        """
        batches_per_file = []

        # Create batch descriptors for sequential chunks from each file
        for file_idx, n_events_limited in enumerate(self.events_per_file_limited):
            # Get file-specific batch size for better mixing
            file_batch_size = self.batch_sizes_per_file[file_idx]

            file_batches = []
            # Split file into batches using file-specific batch size
            # Use events_per_file_limited to respect equal_events constraint
            for start_idx in range(0, n_events_limited, file_batch_size):
                end_idx = min(start_idx + file_batch_size, n_events_limited)

                batch_size = end_idx - start_idx
                file_batches.append(
                    {
                        "file_idx": file_idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "source_file_idx": file_idx,
                        "batch_size": batch_size,
                    }
                )

            batches_per_file.append(file_batches)
            print(f"  File {file_idx}: {len(file_batches)} batches")

        total_batches = sum(len(batches) for batches in batches_per_file)
        print(
            f"Created {total_batches} total batch descriptors across {len(batches_per_file)} files"
        )
        return batches_per_file

    def _read_concatenate_shuffle_batches(
        self, batch_descriptors: list[dict]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Read batches from multiple files, concatenate them, and shuffle.

        Parameters
        ----------
        batch_descriptors : list[dict]
            List of batch descriptors, one per file to read from

        Returns
        -------
        tuple[np.ndarray, dict[str, np.ndarray]]
            Concatenated and shuffled events array and constituent arrays
        """
        all_events = []
        all_constituents = {const_name: [] for const_name in self.constituent_names}

        # Read batches from each file
        for batch_descriptor in batch_descriptors:
            events_batch, constituent_batches = self._read_batch(
                file_idx=batch_descriptor["file_idx"],
                start_idx=batch_descriptor["start_idx"],
                end_idx=batch_descriptor["end_idx"],
            )
            all_events.append(events_batch)
            for const_name in self.constituent_names:
                all_constituents[const_name].append(constituent_batches[const_name])

        # Concatenate all batches
        concatenated_events = np.concatenate(all_events, axis=0)
        concatenated_constituents = {
            const_name: np.concatenate(const_batches, axis=0)
            for const_name, const_batches in all_constituents.items()
        }

        # Shuffle the concatenated batch
        total_size = concatenated_events.shape[0]
        shuffle_order = self.rng.permutation(total_size)

        concatenated_events = concatenated_events[shuffle_order]
        for const_name in self.constituent_names:
            concatenated_constituents[const_name] = concatenated_constituents[
                const_name
            ][shuffle_order]

        return concatenated_events, concatenated_constituents

    def _read_batch(
        self, file_idx: int, start_idx: int, end_idx: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read one sequential event range from one input file."""
        file_handle = self.file_handles[file_idx]
        events_batch = cast(h5py.Dataset, file_handle["events"])[start_idx:end_idx]

        source_file_array = np.full(end_idx - start_idx, file_idx, dtype=np.int32)
        if "source_file_idx" in events_batch.dtype.names:
            # Rewrite existing source_file_idx if it already exists (e.g. from a previous merge)
            events_batch["source_file_idx"] = source_file_array
        else:
            events_batch = append_fields(
                events_batch,
                "source_file_idx",
                source_file_array,
                dtypes=np.int32,
                usemask=False,
            )  # pyright: ignore[reportCallIssue]

        constituent_batches = {
            const_name: cast(h5py.Dataset, file_handle[const_name])[start_idx:end_idx]
            for const_name in self.constituent_names
        }
        return events_batch, constituent_batches

    def _output_path(self, output_index: int) -> Path:
        """Return the output path for a numbered file."""
        if self.events_per_output_file is None:
            return self.output

        suffix = self.output.suffix or ".h5"
        stem = self.output.stem if self.output.suffix else self.output.name
        return self.output.with_name(f"{stem}_{output_index:05d}{suffix}")

    def _batches(self):
        """Yield sequential batches or shuffled mixed-file batches."""
        if self.no_shuffle:
            for file_idx, n_events in enumerate(self.events_per_file_limited):
                for start_idx in range(0, n_events, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_events)
                    yield self._read_batch(file_idx, start_idx, end_idx)
            return

        max_batches = max(len(batches) for batches in self.batch_descriptors)
        for batch_round in range(max_batches):
            batch_descriptors = [
                file_batches[batch_round]
                for file_batches in self.batch_descriptors
                if batch_round < len(file_batches)
            ]
            yield self._read_concatenate_shuffle_batches(batch_descriptors)

    @staticmethod
    def _chunk_shape(key: str, value: np.ndarray):
        if key == "events":
            return (max(1, min(1024, len(value))),)
        return (max(1, min(64, len(value))), value.shape[1])

    def run(self) -> None:
        """
        Execute the merge and shuffle operation.
        """
        mode = "concatenated" if self.no_shuffle else "shuffled"
        if self.events_per_output_file is None:
            print(f"Writing {mode} data to {self.output}...")
        else:
            print(
                f"Writing {mode} data to numbered files based on {self.output} "
                f"(approximately {self.events_per_output_file} events per file)..."
            )

        events_written = 0
        events_in_output = 0
        output_index = 0
        output_paths: list[Path] = []
        out_file: h5py.File | None = None
        events_dataset: h5py.Dataset | None = None
        constituent_datasets: dict[str, h5py.Dataset] = {}
        pbar = tqdm(total=self.total_events, desc="Writing events")

        try:
            for events_batch, constituent_batches in self._batches():
                # Apply event limit if specified
                if self.n_events_limit is not None:
                    remaining = self.n_events_limit - events_written
                    if remaining <= 0:
                        break
                    if events_batch.shape[0] > remaining:
                        events_batch = events_batch[:remaining]
                        for const_name in constituent_batches:
                            constituent_batches[const_name] = constituent_batches[
                                const_name
                            ][:remaining]

                if out_file is None:
                    output_path = self._output_path(output_index)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Opening output file: {output_path}")
                    out_file = h5py.File(output_path, "w")
                    output_paths.append(output_path)
                    events_dataset = None
                    constituent_datasets = {}
                    events_in_output = 0

                if events_dataset is None:
                    # Create datasets on first batch
                    events_dataset = out_file.create_dataset(
                        "events",
                        data=events_batch,
                        maxshape=(None,),
                        chunks=self._chunk_shape("events", events_batch),
                        compression=hdf5plugin.LZ4(),
                    )

                    # Create constituent datasets
                    for const_name, const_data in constituent_batches.items():
                        constituent_datasets[const_name] = out_file.create_dataset(
                            const_name,
                            data=const_data,
                            maxshape=(None,) + const_data.shape[1:],
                            chunks=self._chunk_shape(const_name, const_data),
                            compression=hdf5plugin.LZ4(),
                        )

                    # Write metadata
                    self._write_metadata(out_file, output_index)

                else:
                    # Append to existing datasets
                    events_dataset.resize(
                        (events_dataset.shape[0] + events_batch.shape[0]), axis=0
                    )
                    events_dataset[-events_batch.shape[0] :] = events_batch

                    # Append constituents
                    for const_name, const_data in constituent_batches.items():
                        const_dataset = constituent_datasets[const_name]
                        const_dataset.resize(
                            (const_dataset.shape[0] + const_data.shape[0]), axis=0
                        )
                        const_dataset[-const_data.shape[0] :] = const_data

                batch_size = events_batch.shape[0]
                events_written += batch_size
                events_in_output += batch_size
                pbar.update(batch_size)

                if (
                    self.events_per_output_file is not None
                    and events_in_output >= self.events_per_output_file
                ):
                    out_file.attrs["events_in_file"] = events_in_output
                    out_file.close()
                    out_file = None
                    output_index += 1

                # Stop if we've reached the event limit
                if (
                    self.n_events_limit is not None
                    and events_written >= self.n_events_limit
                ):
                    break
        finally:
            if out_file is not None:
                out_file.attrs["events_in_file"] = events_in_output
                out_file.close()
            pbar.close()

        print(
            f"Successfully wrote {events_written} events to "
            f"{len(output_paths)} output file(s)"
        )

    def _write_metadata(self, h5file: h5py.File, output_index: int) -> None:
        """
        Write metadata as HDF5 attributes.

        Parameters
        ----------
        h5file : h5py.File
            Output HDF5 file handle
        """
        # Convert input file paths to absolute path strings
        source_files_list = [str(f) for f in self.input_files]

        h5file.attrs["source_files"] = json.dumps(source_files_list)
        h5file.attrs["source_file_idx_description"] = (
            "Index into source_files list indicating origin file (0-indexed)"
        )
        h5file.attrs["shuffle_seed"] = (
            str(self.seed) if self.seed is not None else "None"
        )
        h5file.attrs["created_at"] = datetime.now().isoformat()
        h5file.attrs["total_events"] = self.total_events
        h5file.attrs["output_file_index"] = output_index
        h5file.attrs["events_per_file_target"] = (
            self.events_per_output_file
            if self.events_per_output_file is not None
            else self.total_events
        )
        h5file.attrs["created_by"] = "merge_shuffle_hdf5.py"

    def close(self) -> None:
        """Close all input file handles."""
        for f in self.file_handles:
            if f:
                f.close()
        self.file_handles = []

    def __del__(self) -> None:
        """Destructor to ensure file handles are closed."""
        self.close()


if __name__ == "__main__":
    args = get_parser()

    merger = MergeShuffleHDF5(args=args)
    merger.run()
    merger.close()

    print("Done!")
