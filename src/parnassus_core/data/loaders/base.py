"""Abstract base class for file loading backends.

Defines the interface that all file loaders (ROOT, HDF5) must implement.
PhysicsDataset uses this interface to load data in a backend-agnostic way.
"""

from abc import ABC, abstractmethod

import numpy as np


class FileLoader(ABC):
    """Abstract file loading backend.

    Named FileLoader (not DataLoader) to avoid clash with torch.utils.data.DataLoader.

    ROOT returns ragged arrays (variable-length per event).
    HDF5 returns padded arrays with 'valid' mask.
    PhysicsDataset normalizes both to the same internal representation.

    Implementations:
    - ROOTFileLoader: Loads data from ROOT files via uproot
    - HDF5FileLoader: Loads data from HDF5 files via h5py
    """

    def __init__(
        self,
        file_path: str,
        variable_remapping: dict[str, str] | None = None,
    ):
        """Initialize file loader.

        Args:
            file_path: Path to the file to load
            variable_remapping: Optional dict mapping file variable names to
                internal names (e.g., {"Sim_PV_x": "pv_x"} for HDF5)
        """
        self.file_path = file_path
        self.variable_remapping = variable_remapping or {}

    @abstractmethod
    def load_particle_collection(
        self,
        collection_name: str,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load particle-level variables for a collection.

        Args:
            collection_name: Name of collection ("truth", "pflow", "hadrons")
            variables: List of variable names to load (without collection prefix)
            entry_start: First event index to load
            entry_stop: Last event index to load (exclusive)

        Returns:
            Dictionary mapping variable names to numpy arrays.
            For ROOT: ragged arrays (dtype=object, each element is 1D array)
            For HDF5: padded 2D arrays with corresponding valid mask

        Example:
            >>> loader.load_particle_collection("truth", ["pt", "eta"], 0, 100)
            {"pt": array([...]), "eta": array([...])}
        """
        ...

    def load_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load multiple branches in a single bulk read.

        Optional method for backends that support bulk reads of heterogeneous
        branches (e.g. ROOT). Default implementation raises NotImplementedError.

        Args:
            branch_names: Raw branch names as they appear in the file
            entry_start: First event index
            entry_stop: Last event index (exclusive)

        Returns:
            ``{branch_name: array}``
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load_branches()"
        )

    @abstractmethod
    def load_event_variables(
        self,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load event-level variables.

        Args:
            variables: List of variable names to load
            entry_start: First event index to load
            entry_stop: Last event index to load (exclusive)

        Returns:
            Dictionary mapping variable names to 1D numpy arrays

        Example:
            >>> loader.load_event_variables(["eventNumber", "trueNumPU"], 0, 100)
            {"eventNumber": array([...]), "trueNumPU": array([...])}
        """
        ...

    @abstractmethod
    def get_num_particles(
        self,
        collection_name: str,
        entry_start: int,
        entry_stop: int,
    ) -> np.ndarray:
        """Get particle count per event for a collection.

        Args:
            collection_name: Name of collection ("truth", "pflow", "hadrons")
            entry_start: First event index to load
            entry_stop: Last event index to load (exclusive)

        Returns:
            1D array of particle counts per event

        Example:
            >>> loader.get_num_particles("truth", 0, 100)
            array([145, 203, 187, ...])
        """
        ...

    @abstractmethod
    def get_total_events(self) -> int:
        """Get total number of events in the file.

        Returns:
            Total event count
        """
        ...

    @abstractmethod
    def close(self):
        """Close the file and release resources."""
        ...

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
