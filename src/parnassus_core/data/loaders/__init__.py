"""File loader backends for PhysicsDataset.

Provides the abstract FileLoader interface, concrete implementations for the
ROOT, HDF5 and LUND file formats, a selecting wrapper, and
:func:`open_file_loader`, which picks the eager-dataset reader from the path.
"""

from pathlib import Path

from parnassus_core.data.loaders.base import FileLoader
from parnassus_core.data.loaders.filtered_loader import (
    EventSelector,
    FilteredFileLoader,
)
from parnassus_core.data.loaders.hdf5_loader import HDF5FileLoader
from parnassus_core.data.loaders.lund_loader import (
    LUND_SUFFIXES,
    LUNDFileLoader,
    is_lund_path,
    resolve_lund_files,
)
from parnassus_core.data.loaders.root_loader import ROOTFileLoader

__all__ = [
    "LUND_SUFFIXES",
    "EventSelector",
    "FileLoader",
    "FilteredFileLoader",
    "HDF5FileLoader",
    "LUNDFileLoader",
    "ROOTFileLoader",
    "is_lund_path",
    "open_file_loader",
    "resolve_lund_files",
]


def open_file_loader(
    file_path: str | Path,
    *,
    tree_name: str = "evt_tree",
    num_workers: int = 6,
    variable_remapping: dict[str, str] | None = None,
    max_events: int | None = None,
) -> FileLoader:
    """Open the reader for ``file_path``, chosen by its extension.

    Parameters
    ----------
    file_path : str or Path
        A ``.root`` file, or LUND input: a ``.dat``/``.lund`` file, a directory
        of them, or a glob.
    tree_name, num_workers, variable_remapping
        ROOT options (see :class:`ROOTFileLoader`).
    max_events : int or None
        Events the caller will read at most.  The LUND reader stops parsing
        there; the ROOT reader reads only what is requested anyway.

    Returns
    -------
    FileLoader
        A :class:`ROOTFileLoader` or :class:`LUNDFileLoader`.

    Raises
    ------
    ValueError
        If the path is neither ROOT nor LUND input.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".root":
        return ROOTFileLoader(
            str(path),
            tree_name=tree_name,
            num_workers=num_workers,
            variable_remapping=variable_remapping,
        )
    if is_lund_path(path):
        return LUNDFileLoader(path, max_events=max_events)
    raise ValueError(
        f"Unsupported input {file_path}: expected a .root file, or LUND input "
        f"(a {'/'.join(LUND_SUFFIXES)} file, a directory of them, or a glob)."
    )
