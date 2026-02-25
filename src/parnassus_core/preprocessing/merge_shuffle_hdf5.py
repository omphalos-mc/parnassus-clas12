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
        help="Output HDF5 file path.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
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
    Merge and shuffle multiple HDF5 files into a single output file.

    This class reads multiple HDF5 files (outputs from root_to_hdf5.py),
    shuffles all events globally, and writes them to a new HDF5 file
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
        # Discover input files
        self.input_files = discover_input_files(args.input)
        self.output = Path(args.output)
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.n_events_limit = args.nevents
        self.equal_events = args.equal_events

        # Open all input files and store handles
        print("Opening input files...")
        self.file_handles: list[h5py.File] = []
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

        # Create batch descriptors for efficient sequential reading
        print(f"Creating batch descriptors (seed={self.seed})...")
        self.rng = np.random.default_rng(self.seed)
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
            file_idx = batch_descriptor["file_idx"]
            start_idx = batch_descriptor["start_idx"]
            end_idx = batch_descriptor["end_idx"]
            source_file_idx = batch_descriptor["source_file_idx"]

            file_handle = self.file_handles[file_idx]

            # Read sequential batch from file (efficient!)
            events_batch = cast(h5py.Dataset, file_handle["events"])[start_idx:end_idx]

            # Add source_file_idx field to events
            batch_size = end_idx - start_idx
            source_file_array = np.full(batch_size, source_file_idx, dtype=np.int32)
            events_batch = append_fields(
                events_batch,
                "source_file_idx",
                source_file_array,
                dtypes=np.int32,
                usemask=False,
            )  # pyright: ignore[reportCallIssue]

            all_events.append(events_batch)

            # Read constituents
            for const_name in self.constituent_names:
                constituent_batch = cast(h5py.Dataset, file_handle[const_name])[
                    start_idx:end_idx
                ]
                all_constituents[const_name].append(constituent_batch)

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

    def run(self) -> None:
        """
        Execute the merge and shuffle operation.
        """
        print(f"Writing shuffled data to {self.output}...")

        # Determine maximum number of batches across all files
        max_batches = max(len(batches) for batches in self.batch_descriptors)

        events_written = 0
        pbar = tqdm(total=self.total_events, desc="Writing events")

        # Open output file once and keep it open
        with h5py.File(self.output, "w") as out_file:
            events_dataset = None
            constituent_datasets = {}

            # Process batches in rounds: each round reads one batch from each file
            for batch_round in range(max_batches):
                # Collect batch descriptors for this round (one per file, if available)
                current_batch_descriptors = []
                for file_batches in self.batch_descriptors:
                    if batch_round < len(file_batches):
                        current_batch_descriptors.append(file_batches[batch_round])

                # Read from all files, concatenate, and shuffle
                events_batch, constituent_batches = (
                    self._read_concatenate_shuffle_batches(current_batch_descriptors)
                )

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

                if events_dataset is None:
                    # Create datasets on first batch
                    events_dataset = out_file.create_dataset(
                        "events",
                        data=events_batch,
                        maxshape=(None,),
                        chunks=True,
                        compression=hdf5plugin.LZ4(),
                    )

                    # Create constituent datasets
                    for const_name, const_data in constituent_batches.items():
                        constituent_datasets[const_name] = out_file.create_dataset(
                            const_name,
                            data=const_data,
                            maxshape=(None,) + const_data.shape[1:],
                            chunks=True,
                            compression=hdf5plugin.LZ4(),
                        )

                    # Write metadata
                    self._write_metadata(out_file)

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
                pbar.update(batch_size)

                # Stop if we've reached the event limit
                if (
                    self.n_events_limit is not None
                    and events_written >= self.n_events_limit
                ):
                    break

        pbar.close()
        print(f"Successfully wrote {events_written} events to {self.output}")

    def _write_metadata(self, h5file: h5py.File) -> None:
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
