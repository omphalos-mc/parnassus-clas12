"""ROOT file loader via uproot.

Extracts pure file I/O from the legacy ``CMSDataset``. Reads ROOT TTree
branches as numpy arrays — ragged (``dtype=object``) for particle branches,
flat 1-D for event-level branches.

Key design principle: **bulk reads**. Always use ``tree.arrays([...])``
to load multiple branches in a single I/O call, then split afterwards.
Never use ``tree[branch].array()`` one branch at a time.

Usage::

    loader = ROOTFileLoader("data.root", tree_name="evt_tree", num_workers=6)
    truth = loader.load_particle_collection("truth", ["pt", "eta"], 0, 100)
    events = loader.load_event_variables(["eventNumber", "ntruth"], 0, 100)
    loader.close()

    # Or: bulk load everything at once
    all_data = loader.load_branches(
        ["truth_pt", "truth_eta", "pflow_pt", "ntruth", "eventNumber"],
        entry_start=0, entry_stop=100,
    )
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import uproot

from parnassus_core.data.loaders.base import FileLoader


class ROOTFileLoader(FileLoader):
    """Load data from ROOT files via ``uproot``.

    The ROOT file is expected to have a flat TTree where:

    - Particle-level branches are named ``{collection}_{variable}``
      (e.g., ``truth_pt``, ``pflow_d0Error``) and contain ragged arrays.
    - Particle counts are in ``n{collection}`` branches
      (e.g., ``ntruth``, ``npflow``).
    - Event-level branches are scalars (e.g., ``eventNumber``).

    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        TTree name inside the file.
    num_workers : int
        Number of decompression threads for uproot.
    variable_remapping : dict[str, str] | None
        Optional remapping of branch names → internal names.
    """

    def __init__(
        self,
        file_path: str,
        tree_name: str = "evt_tree",
        num_workers: int = 6,
        variable_remapping: dict[str, str] | None = None,
    ):
        super().__init__(file_path, variable_remapping)
        self._file = uproot.open(file_path, num_workers=num_workers)
        self._tree = self._file[tree_name]

    # ------------------------------------------------------------------
    # Core bulk-read method
    # ------------------------------------------------------------------

    def load_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load multiple branches in a single bulk read.

        This is the most efficient way to read from ROOT — all requested
        branches are fetched in one ``tree.arrays()`` call.  Ragged
        (particle-level) and scalar (event-level) branches can be mixed
        in a single call.

        Parameters
        ----------
        branch_names : list[str]
            Branch names exactly as they appear in the TTree
            (e.g., ``["truth_pt", "ntruth", "eventNumber"]``).
        entry_start, entry_stop : int
            Event range ``[entry_start, entry_stop)``.

        Returns
        -------
        dict[str, np.ndarray]
            ``{branch_name: array}`` — ragged branches have
            ``dtype=object``, scalar branches are flat 1-D.
        """
        if not branch_names:
            return {}
        return self._tree.arrays(
            branch_names,
            library="np",
            entry_start=entry_start,
            entry_stop=entry_stop,
        )

    def iter_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int,
        step_size: int,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Bulk-read ``branch_names`` in consecutive chunks of events.

        Same output format as :meth:`load_branches`, one dict per chunk of
        ``step_size`` events (the last chunk may be shorter).  Use this when
        the ragged ``dtype=object`` arrays of the full range would not fit in
        memory at once.

        Parameters
        ----------
        branch_names : list[str]
            Branch names exactly as they appear in the TTree.
        entry_start, entry_stop : int
            Event range ``[entry_start, entry_stop)``.
        step_size : int
            Number of events per chunk.

        Yields
        ------
        dict[str, np.ndarray]
            ``{branch_name: array}`` for one chunk of events.
        """
        if not branch_names or entry_stop <= entry_start:
            return
        yield from self._tree.iterate(
            branch_names,
            library="np",
            entry_start=entry_start,
            entry_stop=entry_stop,
            step_size=step_size,
        )

    # ------------------------------------------------------------------
    # FileLoader interface (convenience wrappers around load_branches)
    # ------------------------------------------------------------------

    def load_particle_collection(
        self,
        collection_name: str,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load ragged particle arrays from ROOT branches.

        Branch names are ``{collection_name}_{variable}``.  The returned
        arrays have ``dtype=object`` — each element is a 1-D numpy array
        of that event's particles.

        Parameters
        ----------
        collection_name : str
            E.g. ``"truth"``, ``"pflow"``.
        variables : list[str]
            Bare variable names (``"pt"``, ``"eta"``).
        entry_start, entry_stop : int
            Event range ``[entry_start, entry_stop)``.

        Returns
        -------
        dict[str, np.ndarray]
            ``{bare_var: ragged_array(dtype=object)}``
        """
        branch_names = [f"{collection_name}_{v}" for v in variables]
        arrays = self.load_branches(branch_names, entry_start, entry_stop)
        return {var: arrays[f"{collection_name}_{var}"] for var in variables}

    def load_event_variables(
        self,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load scalar event-level branches.

        Uses a single bulk ``tree.arrays()`` call for efficiency.

        Parameters
        ----------
        variables : list[str]
            Branch names (e.g. ``["eventNumber"]``).
        entry_start, entry_stop : int
            Event range.

        Returns
        -------
        dict[str, np.ndarray]
            ``{var_name: 1-D array}``
        """
        # Build branch name list, applying remapping
        branch_names: list[str] = []
        name_map: dict[str, str] = {}  # branch_name → internal_name
        for var in variables:
            branch_name = var
            for file_name, internal_name in self.variable_remapping.items():
                if internal_name == var:
                    branch_name = file_name
                    break
            branch_names.append(branch_name)
            name_map[branch_name] = var

        arrays = self.load_branches(branch_names, entry_start, entry_stop)
        return {name_map[bn]: arrays[bn] for bn in branch_names}

    def get_num_particles(
        self,
        collection_name: str,
        entry_start: int,
        entry_stop: int,
    ) -> np.ndarray:
        """Read ``n{collection}`` branch for particle counts.

        Returns
        -------
        np.ndarray
            1-D integer array of particle counts per event.
        """
        branch = f"n{collection_name}"
        arrays = self.load_branches([branch], entry_start, entry_stop)
        return arrays[branch]

    def get_total_events(self) -> int:
        """Total events in the TTree."""
        return self._tree.num_entries

    def close(self):
        """Close the ROOT file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
