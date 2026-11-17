"""LUND text-file loader for generator-level (truth) particles.

LUND is the text format event generators write for the CLAS12 simulation
(https://gemc.jlab.org/gemc/html/documentation/generator/lund.html).  Each event
is a header line whose first field is the number of particle lines that follow,
then one 14-column line per particle::

    index  lifetime  type  pid  parent  daughter  px  py  pz  E  mass  vx  vy  vz

with momenta, energy and mass in GeV and vertices in cm.  The header carries 10
to 100 fields; only the first is used.

The CLAS12 hipo ``MC::Lund`` bank is a row-for-row copy of this file, and the
training truth collection was built from that bank, so the selection here
reproduces it: keep final-state particles (``type == 1``), drop electron
neutrinos, and drop particles with a polar angle below 1 degree.  The loader
exposes the result as the ``truth`` collection, branch for branch like the
ROOT ``evt_tree``.  There is no reconstructed (``pflow``) collection, so LUND
input serves evaluation only.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parnassus_core.data.loaders.base import FileLoader

LUND_SUFFIXES = (".dat", ".lund")
COLLECTION = "truth"
COUNT_BRANCH = f"n{COLLECTION}"
EVENT_NUMBER_BRANCH = "eventNumber"
PARTICLE_VARS = ("p", "theta", "phi", "beta", "vx", "vy", "vz", "pdgId")

N_PARTICLE_COLUMNS = 14
MIN_HEADER_FIELDS = 10
FINAL_STATE_TYPE = 1
ELECTRON_NEUTRINO_PDGID = 12
MIN_THETA_DEG = 1.0

# Zero-based LUND particle-line columns.
_TYPE, _PID = 2, 3
_PX, _PY, _PZ = 6, 7, 8
_MASS = 10
_VX, _VY, _VZ = 11, 12, 13

_GLOB_CHARS = frozenset("*?[")


def is_lund_path(path: str | Path) -> bool:
    """Whether ``path`` names LUND input: a LUND file, a directory, or a glob."""
    path = Path(path)
    return (
        path.suffix.lower() in LUND_SUFFIXES
        or path.is_dir()
        or bool(_GLOB_CHARS & set(str(path)))
    )


def resolve_lund_files(path: str | Path) -> list[Path]:
    """Expand LUND input to the files it names, in read order.

    Parameters
    ----------
    path : str or Path
        A LUND file, a directory (its ``*.dat`` and ``*.lund`` files), or a glob
        pattern.

    Returns
    -------
    list[Path]
        The files, sorted by name for directories and globs.

    Raises
    ------
    FileNotFoundError
        If the path names no file.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in LUND_SUFFIXES
        )
    elif _GLOB_CHARS & set(str(path)):
        files = sorted(Path(match) for match in glob.glob(str(path)))
        files = [file for file in files if file.is_file()]
    else:
        files = [path] if path.is_file() else []
    if not files:
        raise FileNotFoundError(f"No LUND files found at {path}")
    return files


@dataclass(frozen=True)
class LundEvents:
    """Truth particles of consecutive LUND events, as flat arrays.

    Attributes
    ----------
    counts : np.ndarray
        Selected particles per event, ``int32``, shape ``(n_events,)``.  Events
        whose particles were all cut are kept with a count of 0.
    offsets : np.ndarray
        ``[0, cumsum(counts)]``, shape ``(n_events + 1,)``.
    particles : dict[str, np.ndarray]
        ``{var: flat array}`` for every name in :data:`PARTICLE_VARS`.
    """

    counts: np.ndarray
    offsets: np.ndarray
    particles: dict[str, np.ndarray]

    @property
    def n_events(self) -> int:
        return len(self.counts)

    def head(self, n_events: int) -> LundEvents:
        """The first ``n_events`` events."""
        stop = int(self.offsets[n_events])
        return LundEvents(
            counts=self.counts[:n_events],
            offsets=self.offsets[: n_events + 1],
            particles={var: arr[:stop] for var, arr in self.particles.items()},
        )


def _read_particle_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Split a LUND file into events.

    Returns
    -------
    n_rows : np.ndarray
        Particle lines per event, shape ``(n_events,)``.
    table : np.ndarray
        All particle lines, ``float64``, shape ``(sum(n_rows), 14)``.
    """
    lines = path.read_text().splitlines()
    n_rows: list[int] = []
    rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        header = lines[i].split()
        i += 1
        if not header:
            continue
        try:
            declared = float(header[0])
        except ValueError:
            declared = -1.0
        if len(header) < MIN_HEADER_FIELDS or declared < 0 or not declared.is_integer():
            raise ValueError(
                f"{path}:{i}: expected an event header (particle count followed "
                f"by at least {MIN_HEADER_FIELDS - 1} more fields), got "
                f"{lines[i - 1]!r}"
            )
        n_declared = int(declared)
        n_found = 0
        while n_found < n_declared:
            if i >= len(lines):
                raise ValueError(
                    f"{path}:{i}: event truncated: its header declares "
                    f"{n_declared} particles, but the file ends after {n_found}"
                )
            fields = lines[i].split()
            i += 1
            if not fields:
                continue
            if len(fields) != N_PARTICLE_COLUMNS:
                raise ValueError(
                    f"{path}:{i}: expected a {N_PARTICLE_COLUMNS}-column particle "
                    f"line, got {len(fields)} columns: {lines[i - 1]!r}"
                )
            rows.append(fields)
            n_found += 1
        n_rows.append(n_declared)
    table = np.array(rows, dtype=np.float64).reshape(-1, N_PARTICLE_COLUMNS)
    return np.asarray(n_rows, dtype=np.int64), table


def read_lund_file(path: str | Path) -> LundEvents:
    """Read one LUND file into truth particles.

    Kinematics are computed as for the training truth collection:
    ``p = |(px, py, pz)|``, ``theta = atan2(sqrt(px^2 + py^2), pz)``,
    ``phi = atan2(py, px)`` and ``beta = p / sqrt(mass^2 + p^2)`` with the
    file's mass column.  The file values are rounded to float32 first, the
    precision of the hipo ``MC::Lund`` bank, so the 1-degree cut falls on the
    same particles.

    Parameters
    ----------
    path : str or Path
        The LUND file.

    Returns
    -------
    LundEvents
        One entry per event in the file; ``float32`` kinematics, ``int32``
        pdgId.

    Raises
    ------
    ValueError
        On a malformed header or particle line, naming ``file:line``.
    """
    path = Path(path)
    n_rows, table = _read_particle_rows(path)
    table = table.astype(np.float32).astype(np.float64)

    px, py, pz = table[:, _PX], table[:, _PY], table[:, _PZ]
    p = np.sqrt(px**2 + py**2 + pz**2)
    theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = p / np.sqrt(table[:, _MASS] ** 2 + p**2)
    pdg_id = np.rint(table[:, _PID]).astype(np.int32)

    selected = (
        (np.rint(table[:, _TYPE]) == FINAL_STATE_TYPE)
        & (np.abs(pdg_id) != ELECTRON_NEUTRINO_PDGID)
        & (np.degrees(theta) >= MIN_THETA_DEG)
    )
    event_of_row = np.repeat(np.arange(len(n_rows)), n_rows)
    counts = np.bincount(event_of_row[selected], minlength=len(n_rows)).astype(np.int32)
    particles = {
        "p": p[selected],
        "theta": theta[selected],
        "phi": np.arctan2(py, px)[selected],
        "beta": beta[selected],
        "vx": table[selected, _VX],
        "vy": table[selected, _VY],
        "vz": table[selected, _VZ],
    }
    particles = {var: arr.astype(np.float32) for var, arr in particles.items()}
    particles["pdgId"] = pdg_id[selected]
    return LundEvents(
        counts=counts,
        offsets=np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]),
        particles=particles,
    )


def _to_ragged(flat: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-event ``dtype=object`` array of views into ``flat``.

    Filled element by element: assigning a list of equal-length arrays to an
    object array's slice broadcasts it into a 2-D array instead.
    """
    ragged = np.empty(len(counts), dtype=object)
    for index, part in enumerate(np.split(flat, np.cumsum(counts)[:-1])):
        ragged[index] = part
    return ragged


class LUNDFileLoader(FileLoader):
    """Load the truth collection from LUND files.

    Files are parsed lazily, in order, the first time an event in them is
    requested, and kept in memory until :meth:`close`.  ``eventNumber`` is the
    event's position in the concatenated input, starting at 0, counting every
    event in the files (also those left without selected particles).

    Parameters
    ----------
    path : str or Path
        A LUND file, a directory of them, or a glob (see
        :func:`resolve_lund_files`).
    max_events : int or None
        Read at most this many events; ``None`` reads every file.
    """

    def __init__(self, path: str | Path, max_events: int | None = None):
        super().__init__(str(path))
        self._files = resolve_lund_files(path)
        self._max_events = max_events
        self._blocks: list[LundEvents] = []
        self._block_starts: list[int] = []
        self._n_parsed = 0
        self._next_file = 0

    # ------------------------------------------------------------------
    # Lazy parsing
    # ------------------------------------------------------------------

    def _is_exhausted(self) -> bool:
        at_cap = self._max_events is not None and self._n_parsed >= self._max_events
        return at_cap or self._next_file >= len(self._files)

    def _parse_until(self, n_events: int | None) -> None:
        """Parse files until ``n_events`` events are available, or the input ends."""
        while not self._is_exhausted() and (
            n_events is None or self._n_parsed < n_events
        ):
            block = read_lund_file(self._files[self._next_file])
            self._next_file += 1
            if self._max_events is not None:
                block = block.head(
                    min(block.n_events, self._max_events - self._n_parsed)
                )
            self._block_starts.append(self._n_parsed)
            self._blocks.append(block)
            self._n_parsed += block.n_events

    def _slice(
        self, branch_names: list[str], entry_start: int, entry_stop: int
    ) -> dict[str, np.ndarray]:
        """Branches for events ``[entry_start, entry_stop)`` of the parsed input."""
        for name in branch_names:
            if not self.has_branch(name):
                raise KeyError(
                    f"'{name}' is not available from LUND input {self.file_path}: "
                    f"LUND files hold generator-level particles only (the "
                    f"'{COLLECTION}' collection), so they support eval mode only. "
                    f"Available: {sorted(self._branch_names())}"
                )
        entry_stop = max(entry_start, min(entry_stop, self._n_parsed))
        counts_parts: list[np.ndarray] = []
        flat_parts: dict[str, list[np.ndarray]] = {var: [] for var in PARTICLE_VARS}
        for block, block_start in zip(self._blocks, self._block_starts):
            lo = max(entry_start - block_start, 0)
            hi = min(entry_stop - block_start, block.n_events)
            if lo >= hi:
                continue
            counts_parts.append(block.counts[lo:hi])
            p_lo, p_hi = int(block.offsets[lo]), int(block.offsets[hi])
            for var in PARTICLE_VARS:
                flat_parts[var].append(block.particles[var][p_lo:p_hi])

        counts = np.concatenate(counts_parts) if counts_parts else np.zeros(0, np.int32)
        out: dict[str, np.ndarray] = {}
        for name in branch_names:
            if name == COUNT_BRANCH:
                out[name] = counts
            elif name == EVENT_NUMBER_BRANCH:
                out[name] = np.arange(entry_start, entry_stop, dtype=np.int32)
            else:
                var = name.removeprefix(f"{COLLECTION}_")
                flat = (
                    np.concatenate(flat_parts[var])
                    if flat_parts[var]
                    else np.zeros(0, dtype=np.int32 if var == "pdgId" else np.float32)
                )
                out[name] = _to_ragged(flat, counts)
        return out

    @staticmethod
    def _branch_names() -> set[str]:
        return {
            COUNT_BRANCH,
            EVENT_NUMBER_BRANCH,
            *(f"{COLLECTION}_{var}" for var in PARTICLE_VARS),
        }

    # ------------------------------------------------------------------
    # Bulk reads (same output format as ROOTFileLoader)
    # ------------------------------------------------------------------

    def has_branch(self, name: str) -> bool:
        return name in self._branch_names()

    def load_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int,
    ) -> dict[str, np.ndarray]:
        """Load branches for events ``[entry_start, entry_stop)``.

        Particle branches are ragged ``dtype=object`` arrays; ``ntruth`` and
        ``eventNumber`` are flat.
        """
        if not branch_names:
            return {}
        self._parse_until(entry_stop)
        return self._slice(branch_names, entry_start, entry_stop)

    def iter_branches(
        self,
        branch_names: list[str],
        entry_start: int,
        entry_stop: int | None,
        step_size: int,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield :meth:`load_branches` output in chunks of ``step_size`` events.

        ``entry_stop=None`` reads to the end of the input, parsing files only as
        the iteration reaches them.
        """
        if not branch_names:
            return
        lo = entry_start
        while entry_stop is None or lo < entry_stop:
            hi = (
                lo + step_size
                if entry_stop is None
                else min(lo + step_size, entry_stop)
            )
            self._parse_until(hi)
            hi = min(hi, self._n_parsed)
            if hi <= lo:
                return
            yield self._slice(branch_names, lo, hi)
            lo = hi

    # ------------------------------------------------------------------
    # FileLoader interface
    # ------------------------------------------------------------------

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
        """Events in the input (up to ``max_events``); parses every file."""
        self._parse_until(None)
        return self._n_parsed

    def close(self) -> None:
        """Release the parsed events."""
        self._blocks.clear()
        self._block_starts.clear()
