"""Helper utilities for variable name parsing and collection management.

This module provides:
- CollectionInfo: Dataclass for storing collection metadata (backend-agnostic)
- VariableHelper: Instance-based variable name parsing with registered collections

VariableHelper uses registered collection names (not hardcoded) to parse
variable prefixes, extract base names, and derive FeatureScaler keys.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CollectionInfo:
    """Metadata for a particle collection.

    Backend-agnostic: optional fields are set by each dataset class as needed.

    Attributes:
        name: Collection name (e.g., "truth", "pflow", "hadrons")
        n_particles: Array of particle counts per event (eager mode only)
        cumsum: Cumulative sum for flat array indexing (eager mode only)
    """

    name: str
    n_particles: np.ndarray | None = None
    cumsum: np.ndarray | None = None


class VariableHelper:
    """Variable name parsing with registered collections.

    Infers collection membership from variable prefixes. No hardcoded
    collection names — collections are registered at construction time.

    Args:
        collection_names: List of collection names to recognize
            (e.g., ``["truth", "pflow", "hadrons"]``)

    Examples:
        >>> vh = VariableHelper(["truth", "pflow"])
        >>> vh.get_collection("truth_pt")
        'truth'
        >>> vh.get_collection("ht")  # No prefix → None
        >>> vh.get_base_name("pflow_d0My")
        'd0My'
        >>> vh.scaler_key("pflow_d0My")
        'd0'
    """

    def __init__(self, collection_names: list[str]):
        self.collection_names = list(collection_names)
        self._prefixes = {name: f"{name}_" for name in self.collection_names}

    def get_collection(self, var_name: str) -> str | None:
        """Extract collection name from variable.

        Args:
            var_name: Variable name, e.g. ``'truth_pt'``, ``'pflow_d0My'``

        Returns:
            Collection name (``'truth'``, ``'pflow'``, etc.) or ``None``
            if no recognized prefix matches.

        Examples:
            >>> vh = VariableHelper(["truth", "pflow"])
            >>> vh.get_collection("truth_pt")
            'truth'
            >>> vh.get_collection("pflow_d0My")
            'pflow'
            >>> vh.get_collection("ht")
            None
        """
        for name, prefix in self._prefixes.items():
            if var_name.startswith(prefix):
                return name
        return self.is_npart_var(var_name)

    def get_base_name(self, var_name: str) -> str:
        """Remove collection prefix from variable name.

        Args:
            var_name: Variable name with optional prefix

        Returns:
            Base variable name without collection prefix

        Examples:
            >>> vh = VariableHelper(["truth", "pflow"])
            >>> vh.get_base_name("truth_pt")
            'pt'
            >>> vh.get_base_name("pflow_d0My")
            'd0My'
            >>> vh.get_base_name("ht")
            'ht'
        """
        for prefix in self._prefixes.values():
            if var_name.startswith(prefix):
                return var_name[len(prefix) :]

        if self.is_npart_var(var_name):
            return "npart"

        return var_name

    def scaler_key(self, var_name: str) -> str:
        """Generate FeatureScaler lookup key from variable name.

        Implements 3-step transformation:
        1. Remove collection prefix (truth_, pflow_, etc.)
        2. Replace ``n{collection}`` → ``npart`` for any registered collection
        3. Remove "My" suffix (for d0My, z0My computed IP params)

        Args:
            var_name: Variable name to transform

        Returns:
            Scaler key for FeatureScaler lookup

        Examples:
            >>> vh = VariableHelper(["truth", "pflow"])
            >>> vh.scaler_key("truth_pt")
            'pt'
            >>> vh.scaler_key("pflow_d0My")
            'd0'
            >>> vh.scaler_key("npflow")
            'npart'
            >>> vh.scaler_key("ntruth")
            'npart'
        """
        # Step 1: Remove collection prefix
        key = self.get_base_name(var_name)

        # Step 2: Normalize particle count variable names (n{collection} → npart)
        for coll_name in self.collection_names:
            if key == f"n{coll_name}":
                key = "npart"
                break

        # Step 3: Remove "My" suffix (recalculated IP params share scalers)
        key = key.replace("My", "")

        return key

    def is_npart_var(self, var_name: str) -> str | None:
        """Check if variable is a particle-count variable (n{collection}).

        Returns the collection name if it matches ``n{collection}`` for any
        registered collection, or ``None`` otherwise.

        Examples:
            >>> vh = VariableHelper(["truth", "pflow"])
            >>> vh.is_npart_var("ntruth")
            'truth'
            >>> vh.is_npart_var("npflow")
            'pflow'
            >>> vh.is_npart_var("ht")
        """
        for coll_name in self.collection_names:
            if var_name == f"n{coll_name}":
                return coll_name
        return None
