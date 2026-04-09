"""HDF5 file loader via h5py.

Reads structured HDF5 files produced by ``root_to_hdf5.py``.  Particle
data is stored as padded 2-D arrays ``(N_events, max_particles)`` with a
``valid`` boolean field distinguishing real particles from padding.

Event-level scalars live in a separate ``events`` dataset.

The file handle is opened **lazily** on first read, making the loader
pickleable and safe for PyTorch ``DataLoader`` with ``num_workers > 0``.
Each worker process opens its own independent handle.

Uses ``read_direct`` with pre-allocated reusable buffers for efficient I/O.

Usage::

    loader = HDF5FileLoader("data.h5")
    raw, mask = loader.read_collection(slice(0, 64), "truth")
    events   = loader.read_events(slice(0, 64))
    loader.close()
"""

from __future__ import annotations

import os
from typing import cast

import h5py
import hdf5plugin  # noqa: F401  — registers HDF5 decompression codecs
import numpy as np

from parnassus_core.data.loaders.base import FileLoader


def _maybe_copy(src: np.ndarray) -> np.ndarray:
    """Always return a contiguous array owned by this process (for safety with multi-threading)."""
    return np.nan_to_num(src.copy())


class HDF5FileLoader(FileLoader):
    """Load data from HDF5 files produced by ``root_to_hdf5.py``.

    File layout expected::

        events   (N,)       — structured array with event-level scalars
        truth    (N, 400)   — structured array with particle fields + ``valid``
        pflow    (N, 400)   — structured array with particle fields + ``valid``

    The file handle is opened lazily on first read so the loader is
    pickleable and each ``DataLoader`` worker gets its own handle.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    variable_remapping : dict[str, str] | None
        Optional mapping from file field names → internal names
        (e.g. ``{"Sim_PV_x": "pv_x"}``).
    """

    def __init__(
        self,
        file_path: str,
        variable_remapping: dict[str, str] | None = None,
    ):
        super().__init__(file_path, variable_remapping)
        # Read metadata (shapes, dtypes) then close immediately.
        # The handle will be re-opened lazily on first read.
        with h5py.File(file_path, "r") as f:
            self._meta: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
            for key in f:
                ds = cast(h5py.Dataset, f[key])
                self._meta[str(key)] = (ds.shape, ds.dtype)
        self._file: h5py.File | None = None
        self._pid: int | None = None
        self._dss: dict[str, h5py.Dataset] = {}

    def _ensure_open(self) -> None:
        """Open the HDF5 file if not already open (lazy / per-worker)."""
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None and self._file.id.valid:  # type: ignore
                self._file.close()

            self._file = h5py.File(self.file_path, "r", swmr=True, libver="latest")
            self._pid = pid
            self._dss = {}
            for key in self._file:  # type: ignore
                key_s = str(key)
                self._dss[key_s] = cast(h5py.Dataset, self._file[key_s])

    # ------------------------------------------------------------------
    # Primary read helpers (used by StreamingDataset)
    # ------------------------------------------------------------------

    def read_collection(
        self,
        batch_idx: slice,
        collection_name: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Read a batch of particle data via ``read_direct``.

        Parameters
        ----------
        batch_idx : slice
            Event range (e.g. ``slice(0, 64)``).
        collection_name : str
            HDF5 dataset name (``"truth"``, ``"pflow"``, …).

        Returns
        -------
        data : dict[str, np.ndarray]
            ``{field_name: array(B, max_particles)}`` for every field
            except ``"valid"``.  Arrays are float32.
        valid : np.ndarray
            Boolean mask ``(B, max_particles)`` from the ``valid`` field.
        """
        self._ensure_open()
        ds = self._dss[collection_name]
        n = batch_idx.stop - batch_idx.start
        buf = np.zeros((n, *ds.shape[1:]), dtype=ds.dtype)
        ds.read_direct(buf, batch_idx)

        data: dict[str, np.ndarray] = {}
        valid = _maybe_copy(buf["valid"])
        for field in buf.dtype.names:  # type: ignore[union-attr]
            if field == "valid":
                continue
            data[field] = _maybe_copy(buf[field])
        return data, valid

    def read_events(
        self,
        batch_idx: slice,
    ) -> dict[str, np.ndarray]:
        """Read a batch of event-level scalars via ``read_direct``.

        Parameters
        ----------
        batch_idx : slice
            Event range.

        Returns
        -------
        dict[str, np.ndarray]
            ``{field_name: 1-D array(B,)}`` for every field in the
            ``events`` dataset.  Remapped names are applied if
            ``variable_remapping`` was provided.
        """
        self._ensure_open()
        ds = self._dss["events"]
        n = batch_idx.stop - batch_idx.start
        buf = np.zeros((n,), dtype=ds.dtype)
        ds.read_direct(buf, batch_idx)

        result: dict[str, np.ndarray] = {}
        for field in buf.dtype.names:  # type: ignore[union-attr]
            internal = self.variable_remapping.get(field, field)
            result[internal] = _maybe_copy(buf[field])
        return result

    # ------------------------------------------------------------------
    # FileLoader ABC (kept for compatibility / generic code)
    # ------------------------------------------------------------------

    def load_particle_collection(
        self,
        collection_name: str,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load particle arrays + valid mask (ABC implementation).

        Prefer :meth:`read_collection` for streaming usage.
        """
        data, valid = self.read_collection(
            slice(entry_start, entry_stop),
            collection_name,
        )
        result = {v: data[v] for v in variables if v in data}
        result["valid"] = valid
        return result

    def load_event_variables(
        self,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load event-level variables (ABC implementation).

        Prefer :meth:`read_events` for streaming usage.
        """
        all_data = self.read_events(slice(entry_start, entry_stop))
        return {v: all_data[v] for v in variables if v in all_data}

    def get_num_particles(
        self,
        collection_name: str,
        entry_start: int,
        entry_stop: int,
    ) -> np.ndarray:
        """Count valid particles per event."""
        _, valid = self.read_collection(
            slice(entry_start, entry_stop),
            collection_name,
        )
        return valid.sum(axis=1).astype(np.int64)

    def get_total_events(self) -> int:
        """Total number of events in the file."""
        return int(self._meta["events"][0][0])

    def get_max_particles(self, collection_name: str) -> int:
        """Maximum particle count (padding dimension)."""
        return int(self._meta[collection_name][0][1])

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dss.clear()

    def __getstate__(self) -> dict:
        """Pickle support: drop the open file handle."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dss"] = {}
        return state
