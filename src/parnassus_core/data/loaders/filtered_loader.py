"""Event selection applied while reading, as a wrapper around another loader.

``FilteredFileLoader`` makes a cheap first pass over the few branches an
:class:`EventSelector` needs, remembers which events it keeps, and from then on
serves every read as if the file held only those events.  ``EagerDataset``
therefore materializes the selected events only, and an event cap means
"selected events", so the pass can stop as soon as the cap is reached.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np

from parnassus_core.data.loaders.base import FileLoader


class EventSelector(Protocol):
    """Per-event keep/drop decision over chunks of raw branches.

    Attributes
    ----------
    branches : Sequence[str]
        Branches every chunk passed to the selector must contain.
    optional_branches : Sequence[str]
        Branches added to the chunk when the file has them.
    """

    branches: Sequence[str]
    optional_branches: Sequence[str]

    def __call__(self, chunk: dict[str, np.ndarray]) -> np.ndarray:
        """Boolean keep mask, one entry per event of ``chunk``.

        Called on consecutive chunks in file order, so a stateful selector
        (e.g. one drawing random numbers) sees the events in a fixed order.
        """
        ...


class FilteredFileLoader(FileLoader):
    """Present only the events an :class:`EventSelector` keeps.

    Event indices of every read refer to the kept events: kept event ``i`` is
    the ``i``-th kept event in file order.  Per-event branches (including
    ``eventNumber``) are passed through unchanged.

    Parameters
    ----------
    inner : FileLoader
        Loader of the full file; must implement ``iter_branches`` and
        ``has_branch``.
    selector : EventSelector
        The selection.
    max_kept : int or None
        Stop selecting once this many events are kept; ``None`` scans the whole
        file.
    chunk_size : int
        Events per chunk of the selection pass.
    """

    def __init__(
        self,
        inner: FileLoader,
        selector: EventSelector,
        max_kept: int | None = None,
        chunk_size: int = 200_000,
    ):
        super().__init__(inner.file_path, inner.variable_remapping)
        self._inner = inner
        self._kept = self._select(selector, max_kept, chunk_size)

    def _select(
        self, selector: EventSelector, max_kept: int | None, chunk_size: int
    ) -> np.ndarray:
        """File indices of the kept events, ascending."""
        branches = list(selector.branches) + [
            name for name in selector.optional_branches if self._inner.has_branch(name)
        ]
        kept_parts: list[np.ndarray] = []
        n_kept = 0
        chunk_start = 0
        chunks = self._inner.iter_branches(branches, 0, None, chunk_size)
        try:
            for chunk in chunks:
                n_chunk = len(chunk[branches[0]])
                keep = np.asarray(selector(chunk), dtype=bool)
                if keep.shape != (n_chunk,):
                    raise ValueError(
                        f"Event selector returned a mask of shape {keep.shape} "
                        f"for a chunk of {n_chunk} events."
                    )
                kept_parts.append(chunk_start + np.flatnonzero(keep))
                n_kept += len(kept_parts[-1])
                chunk_start += n_chunk
                if max_kept is not None and n_kept >= max_kept:
                    break
        finally:
            # Stop a lazy reader from parsing further input.
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        kept = np.concatenate(kept_parts) if kept_parts else np.zeros(0, dtype=np.int64)
        return kept[:max_kept] if max_kept is not None else kept

    def has_branch(self, name: str) -> bool:
        return self._inner.has_branch(name)

    def load_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load kept events ``[entry_start, entry_stop)``."""
        if not branch_names:
            return {}
        indices = self._kept[entry_start:entry_stop]
        if not len(indices):
            return self._inner.load_branches(branch_names, 0, 0)
        raw_start = int(indices[0])
        raw = self._inner.load_branches(branch_names, raw_start, int(indices[-1]) + 1)
        return {name: arr[indices - raw_start] for name, arr in raw.items()}

    def iter_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int | None,
        step_size: int,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Stream kept events ``[entry_start, entry_stop)``.

        Reads the file in chunks of ``step_size`` file events, so chunks hold
        at most ``step_size`` kept events; empty chunks are skipped.
        """
        indices = self._kept[entry_start:entry_stop]
        if not branch_names or not len(indices):
            return
        raw_start = int(indices[0])
        chunk_start = raw_start
        position = 0
        for chunk in self._inner.iter_branches(
            branch_names, raw_start, int(indices[-1]) + 1, step_size
        ):
            chunk_stop = chunk_start + len(chunk[branch_names[0]])
            end = int(np.searchsorted(indices, chunk_stop, side="left"))
            local = indices[position:end] - chunk_start
            if len(local):
                yield {name: arr[local] for name, arr in chunk.items()}
            position, chunk_start = end, chunk_stop

    def load_particle_collection(
        self,
        collection_name: str,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        names = [f"{collection_name}_{var}" for var in variables]
        arrays = self.load_branches(names, entry_start, entry_stop)
        return {var: arrays[name] for var, name in zip(variables, names)}

    def load_event_variables(
        self,
        variables: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        return self.load_branches(variables, entry_start, entry_stop)

    def get_num_particles(
        self,
        collection_name: str,
        entry_start: int,
        entry_stop: int,
    ) -> np.ndarray:
        name = f"n{collection_name}"
        return self.load_branches([name], entry_start, entry_stop)[name]

    def get_total_events(self) -> int:
        """Number of kept events."""
        return len(self._kept)

    def close(self) -> None:
        self._inner.close()
