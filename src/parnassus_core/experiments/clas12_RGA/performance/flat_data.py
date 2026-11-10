"""Flat-array load/process backend for the CLAS12 RGA performance plots.

The shared ``parnassus_core.performance`` loader reads every jagged branch with
``library="np"``, which turns each column into an ``ndarray(dtype=object)`` holding one
small ndarray per event. On the 2.4M-event CLAS12 eval files that is ~90M live Python
objects (~64 min and ~21 GB just to load), and every subsequent step -- ``calc_met_ht``,
``calc_particle_residuals`` -- then walks those objects one event at a time.

This module stores the same data as **flat contents + per-event counts/offsets**, read
with ``library="ak"`` (zero-copy flatten). That is exactly the representation
``PlotHelper.plot_features`` reduces to anyway -- its
``if value.dtype == np.object_: value = np.concatenate(value)`` guard simply does not
fire on a flat array -- so the shared plotting path works unchanged.

Measured on ``postprocess_unionRef_2435k_test_eta49Eval.root`` (2,435,107 events):
load+flatten 249 s / 2.6 GB peak, event aggregates 1.8 s, particle residuals 1.5 s.

Nothing here touches shared code: the classes subclass ``Data`` / ``SampleData`` and
override only the methods whose per-event loops are the bottleneck, plus the two
residual methods. CLAS12 residuals are all absolute (reco - truth) except the DIS
kinematics, which stay relative; the shared code instead makes ``pt`` and ``ht``
relative.
"""

from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import awkward as ak
import numpy as np
import uproot

from parnassus_core.data.utils.computed_vars import ComputedVariables
from parnassus_core.performance.config import PflowDataConfig, SampleT, Source
from parnassus_core.performance.performance import (
    DIS_VARS,
    Data,
    SampleData,
    run_once,
)

# Per-particle fields worth reading. The postprocessed CLAS12 files also carry
# px/py/pz/E and a per-event hung_cost_all that nothing downstream consumes -- on the
# union file that is ~590 MB compressed the shared loader reads and then discards.
PARTICLE_FIELDS = (
    "pt",
    "p",  # files carry p or pt; from_flat derives whichever is missing
    "eta",  # absent on CLAS12 (theta is the variable of record); kept for other files
    "theta",
    "phi",
    "vx",
    "vy",
    "vz",
    "beta",  # absent from some older samples
    "class",
    "pdgId",
    "d0",
    "d0Error",
    "z0",
    "z0Error",
)

# Matching indices, written per-particle and padded with -1 by postprocess_eval.
MATCH_FIELDS = ("tr_idx_all", "pf_idx_all")

# Residual variables, in the order calc_particle_residuals uses. Whichever of
# eta/theta (or vx/vy/beta) the file does not carry is dropped.
RESIDUAL_VARS = ("pt", "p", "eta", "theta", "phi", "vx", "vy", "vz", "beta")

COLLECTIONS = ("pflow", "truth", "fastsim")


def _offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _ragged_gather_index(
    offsets: np.ndarray, keep_events: np.ndarray, new_counts: np.ndarray
) -> np.ndarray:
    """Flat positions of every element belonging to ``keep_events``.

    Built as ``event_start + within_event_rank`` so no Python loop is needed.
    """
    new_offsets = _offsets_from_counts(new_counts)
    rank = np.arange(new_offsets[-1], dtype=np.int64) - np.repeat(
        new_offsets[:-1], new_counts
    )
    return np.repeat(offsets[keep_events], new_counts) + rank


def _segment_sum(
    values: np.ndarray, counts: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Per-event sum of a flat array, accumulated in float64.

    ``np.add.reduceat`` returns ``values[i]`` -- not 0 -- for a zero-length segment, so
    empty events have to be masked out and left at 0. That 0 matches what the shared
    ``calc_met_ht`` produces (``np.sum([]) == 0.0``), which the notebook's
    ``drop_empty_events`` switch keys off.
    """
    out = np.zeros(len(counts), dtype=np.float64)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.add.reduceat(
            np.asarray(values, dtype=np.float64), offsets[:-1][nonempty]
        )
    return out


def _collection_field_names(keys: set[str], collection: str) -> list[str]:
    return [
        var
        for var in (*PARTICLE_FIELDS, *MATCH_FIELDS)
        if f"{collection}.{var}" in keys
    ]


def _read_event_scalars(tree, collection: str, keys: set[str], entry_stop) -> dict:
    """Flat per-event DIS branches. Not jagged, so library="np" is correct and cheap."""
    return {
        f"evt_{var}": tree[f"{collection}_{var}"].array(
            library="np", entry_stop=entry_stop
        )
        for var in DIS_VARS
        if f"{collection}_{var}" in keys
    }


def read_clas12_flat(
    path: str | Path,
    entry_stop: int | None = None,
    load_reference: bool = True,
    n_workers: int = 1,
) -> dict:
    """Read a postprocessed CLAS12 eval file into flat per-collection arrays.

    Each collection maps to a dict of flat arrays plus ``counts`` (particles per event)
    and ``offsets`` (``cumsum``, length ``n_events + 1``).

    ``load_reference=False`` skips the ``truth``/``pflow`` collections, mirroring the
    shared loader's ``load_cms`` flag -- they only need reading from the first file of a
    multi-file sample.

    ``n_workers > 1`` reads fields concurrently, each thread using its own
    ``uproot.open`` handle. Sharing one tree object across threads is NOT safe here: it
    silently returns truncated arrays with no error. Even done correctly the win is
    modest (1.7x at 4 workers) and peak RSS roughly triples, so the default is serial.

    ``tree.arrays([...])`` is deliberately not used for the nested sub-branches: its
    expression parser reads the "." in "fastsim.pt" as attribute access and silently
    returns nothing for those names. ``tree[name].array()`` resolves both layouts, so
    fields are read one at a time.
    """
    path = str(path)
    wanted = COLLECTIONS if load_reference else ("fastsim",)

    with uproot.open(path, object_cache=None, array_cache=None) as f:
        tree = f["evt_tree"]
        keys = set(tree.keys())
        jobs = [
            (collection, var)
            for collection in wanted
            for var in _collection_field_names(keys, collection)
        ]

        if n_workers > 1:
            flats = _read_fields_threaded(path, jobs, entry_stop, n_workers)
        else:
            flats = {}
            for job in jobs:
                collection, var = job
                arr = tree[f"{collection}.{var}"].array(
                    library="ak", entry_stop=entry_stop
                )
                flats[job] = _flatten(arr)
                del arr

        data: dict = {
            "eventNumber": tree["eventNumber"].array(
                library="np", entry_stop=entry_stop
            )
        }
        for collection in wanted:
            names = _collection_field_names(keys, collection)
            if not names:
                continue
            entry: dict = {}
            for var in names:
                counts, flat = flats.pop((collection, var))
                # Every field of a collection shares the jagged structure, so the
                # counts only have to be kept from the first one.
                entry.setdefault("counts", counts)
                entry[var] = flat
            entry["offsets"] = _offsets_from_counts(entry["counts"])
            entry.update(_read_event_scalars(tree, collection, keys, entry_stop))
            data[collection] = entry
    return data


def _flatten(arr) -> tuple[np.ndarray, np.ndarray]:
    counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)
    return counts, np.ascontiguousarray(ak.to_numpy(ak.flatten(arr, axis=1)))


def _read_fields_threaded(
    path: str, jobs: list[tuple[str, str]], entry_stop, n_workers: int
) -> dict:
    local = threading.local()

    def read_one(job):
        if not hasattr(local, "tree"):
            local.file = uproot.open(path, object_cache=None, array_cache=None)
            local.tree = local.file["evt_tree"]
        collection, var = job
        arr = local.tree[f"{collection}.{var}"].array(
            library="ak", entry_stop=entry_stop
        )
        return job, _flatten(arr)

    with ThreadPoolExecutor(n_workers) as pool:
        return dict(pool.map(read_one, jobs))


@dataclass
class Clas12FlatData(Data):
    """``Data`` holding flat particle arrays instead of ``dtype=object`` ones.

    ``particle_data_all`` maps a variable name to a single contiguous array of every
    particle in the sample; ``part_counts``/``part_offsets`` carve it into events.
    ``tr_idx``/``pf_idx`` are the *local* (within-event) indices of the valid matched
    pairs, flattened the same way and described by ``match_counts``.

    The ``run_once`` flag names and prerequisites are inherited unchanged, so
    ``SampleData.process_data`` and ``Performance.process`` drive this exactly as they
    drive the shared ``Data``. Build one with :meth:`from_flat`; the inherited
    ``load_from_dict`` would leave ``part_counts`` unset.
    """

    part_counts: np.ndarray | None = field(default=None)
    part_offsets: np.ndarray | None = field(default=None)
    match_counts: np.ndarray | None = field(default=None)

    @classmethod
    def from_flat(
        cls,
        name: str,
        source: Source,
        collection_data: dict,
        event_number: np.ndarray,
        label: str | None = None,
    ) -> Self:
        counts = collection_data["counts"]
        offsets = collection_data["offsets"]

        particle_data = {
            var: collection_data[var]
            for var in PARTICLE_FIELDS
            if var in collection_data
        }
        # Event aggregates need pt and the CLAS12 plots need p, but a file carries only
        # one of them (theta in radians), so derive the other as postprocess_eval does.
        if "theta" in particle_data:
            sin_theta = np.sin(np.asarray(particle_data["theta"], dtype=np.float64))
            if "pt" not in particle_data and "p" in particle_data:
                particle_data["pt"] = particle_data["p"] * sin_theta
            elif "p" not in particle_data and "pt" in particle_data:
                particle_data["p"] = particle_data["pt"] / sin_theta
        event_data = {
            var: collection_data[f"evt_{var}"]
            for var in DIS_VARS
            if f"evt_{var}" in collection_data
        }

        match_counts = None
        if all(var in collection_data for var in MATCH_FIELDS):
            tr_idx, pf_idx = (collection_data[var] for var in MATCH_FIELDS)
            # postprocess pads unmatched slots with -1 on both columns; the survivors
            # stay paired positionally, which is what the shared loader's separate
            # `el[el > -1]` filters relied on too.
            valid = (tr_idx > -1) & (pf_idx > -1)
            particle_data["tr_idx"] = tr_idx[valid]
            particle_data["pf_idx"] = pf_idx[valid]
            match_counts = _segment_sum(valid, counts, offsets).astype(np.int64)

        return cls(
            name=name,
            source=source,
            label=label,
            event_number=event_number,
            event_data=event_data,
            jet_data={},
            particle_data_all=particle_data,
            particle_data_jet={},
            particle_data_bkg={},
            part_counts=counts,
            part_offsets=offsets,
            match_counts=match_counts,
        )

    @property
    def num_events(self) -> int:
        return len(self.part_counts)

    def particle_event_index(self) -> np.ndarray:
        """Event index of every particle, i.e. ``[0, 0, 0, 1, 2, 2, ...]``."""
        return np.repeat(np.arange(self.num_events), self.part_counts)

    def global_match_indices(self, other: Self) -> tuple[np.ndarray, np.ndarray]:
        """Flat positions of the matched ``(self, other)`` particle pairs.

        Turns the stored per-event local indices into positions in the flat arrays by
        adding each side's event offset. Also applies the shared code's
        ``len(indices[i]) > len(data["pt"][i])`` sanity guard, vectorized to a per-event
        mask (it drops nothing on the current files, but keeps parity).
        """
        keep_event = self.match_counts <= self.part_counts
        match_event = np.repeat(np.arange(self.num_events), self.match_counts)
        keep = keep_event[match_event]
        match_event = match_event[keep]
        self_idx = (
            self.part_offsets[match_event] + self.particle_data_all["pf_idx"][keep]
        )
        other_idx = (
            other.part_offsets[match_event] + self.particle_data_all["tr_idx"][keep]
        )
        return self_idx, other_idx

    @run_once("_events_filtered", "events already filtered")
    def filter_events(self, good_events: np.ndarray):
        print(f"Filtering events for {self.name} with source {self.source}")
        order = np.argsort(self.event_number)
        keep = order[np.isin(self.event_number[order], good_events)]

        new_counts = self.part_counts[keep]
        part_keep = _ragged_gather_index(self.part_offsets, keep, new_counts)

        match_keep = None
        if self.match_counts is not None:
            match_offsets = _offsets_from_counts(self.match_counts)
            new_match_counts = self.match_counts[keep]
            match_keep = _ragged_gather_index(match_offsets, keep, new_match_counts)
            self.match_counts = new_match_counts

        for var, value in self.particle_data_all.items():
            index = match_keep if var in {"tr_idx", "pf_idx"} else part_keep
            self.particle_data_all[var] = value[index]
        for var, value in self.event_data.items():
            self.event_data[var] = value[keep]

        self.part_counts = new_counts
        self.part_offsets = _offsets_from_counts(new_counts)
        self.event_number = self.event_number[keep]
        print(
            f"Data(name={self.name}, source={self.source}), "
            f"events filtered: {len(good_events)}"
        )

    @run_once(
        "_event_data_calculated",
        "event data already calculated",
        required_flags=["_events_filtered"],
    )
    def calculate_event_data(self):
        print(f"Calculating event data for {self.name} with source {self.source}")
        counts, offsets = self.part_counts, self.part_offsets
        # No jets are clustered for CLAS12. The shared code leaves njet as a length-0
        # array here, which makes njet_res empty while its siblings are length-N; a
        # zeros array is the well-formed equivalent. njet is not in
        # Clas12RGAPlotHelper.event_vars, so nothing plotted changes.
        self.event_data["njet"] = np.zeros(len(counts), dtype=int)
        self.event_data["npart"] = counts.astype(int)

        pt = np.asarray(self.particle_data_all["pt"], dtype=np.float64)
        phi = np.asarray(self.particle_data_all["phi"], dtype=np.float64)
        self.event_data["met_x"] = _segment_sum(pt * np.cos(phi), counts, offsets)
        self.event_data["met_y"] = _segment_sum(pt * np.sin(phi), counts, offsets)
        self.event_data["ht"] = _segment_sum(pt, counts, offsets)

        # p analogues the event model is trained on: the scalar sum of p, and the
        # hadronic (non-electron) px/py sums, via the same function training uses.
        if "p" in self.particle_data_all:
            p = np.asarray(self.particle_data_all["p"], dtype=np.float64)
            self.event_data["ht_p"] = _segment_sum(p, counts, offsets)
        if "pdgId" in self.particle_data_all:
            # Fastsim and truth store pdgId as float.
            pdgid = np.rint(self.particle_data_all["pdgId"]).astype(int)
            self.event_data.update(
                ComputedVariables.compute_flat_hadronic_missing_p(
                    pdgid, pt, phi, counts, offsets
                )
            )

    @run_once(
        "_event_residuals_calculated",
        "event residuals already calculated",
        required_flags=["_event_data_calculated"],
    )
    def calculate_event_residuals(self, other: Self):
        """Absolute event residuals, except the DIS kinematics, which stay relative.

        Relative residuals of the event aggregates are dominated by events whose truth
        value is near 0 (``Pxh``/``Pyh`` are centred on it), and every event with an
        empty reco sum lands exactly on -1.
        """
        print(
            f"Calculating event residuals for {self.name} with source {self.source}"
            f" against {other.name} with source {other.source}"
        )
        assert self.source != other.source, (
            "Sources must be different for residuals calculation."
        )
        assert (self.source != Source.TR) and (other.source == Source.TR), (
            "Only truth data can be used for residuals calculation."
        )
        keys = [key for key in self.event_data if "_res" not in key]
        for key in keys:
            assert key in other.event_data, f"Key {key} not found in other data."
            res = self.event_data[key] - other.event_data[key]
            if key in DIS_VARS:
                res = res / (other.event_data[key] + 1e-8)
            self.event_data[f"{key}_res"] = res

    @run_once(
        "_particles_residuals_calculated",
        "particle residuals already calculated",
        required_flags=["_events_filtered"],
    )
    def calculate_particle_residuals(self, other: Self):
        print(
            f"Calculating particle residuals for {self.name} with source {self.source}"
            f" against {other.name} with source {other.source}"
        )
        assert self.source != other.source, (
            "Sources must be different for particle residuals calculation."
        )
        assert (self.source != Source.TR) and (other.source == Source.TR), (
            "Only truth data can be used for particle residuals calculation."
        )
        self_idx, other_idx = self.global_match_indices(other)
        for var in RESIDUAL_VARS:
            if var not in self.particle_data_all or var not in other.particle_data_all:
                continue
            reco = np.asarray(self.particle_data_all[var], dtype=np.float64)[self_idx]
            truth = np.asarray(other.particle_data_all[var], dtype=np.float64)[
                other_idx
            ]
            res = reco - truth
            if var == "phi":
                res = np.arctan2(np.sin(res), np.cos(res))
            self.particle_data_all[f"{var}_res"] = res

    @run_once(
        "_impact_paramters_calculated",
        "impact parameters already calculated",
        required_flags=["_events_filtered"],
    )
    def calculate_impact_parameters(self):
        print(
            f"Calculating impact parameters for {self.name} with source {self.source}"
        )
        data = self.particle_data_all
        if "d0" not in data:
            print(f"No impact parameters found in {self.name}. Skipping.")
            return
        d0, z0 = data["d0"], data["z0"]
        d0_error, z0_error = data["d0Error"], data["z0Error"]
        data["d0Sig"] = d0 / (d0_error + 1e-8)
        data["z0Sig"] = z0 / (z0_error + 1e-8)
        data["ip3d"] = np.sqrt(d0**2 + z0**2)
        data["ip3dError"] = np.sqrt(d0_error**2 + z0_error**2)
        data["ip3dSig"] = data["ip3d"] / (data["ip3dError"] + 1e-8)


@dataclass
class Clas12FlatSampleData(SampleData[SampleT]):
    """``SampleData`` that loads via :func:`read_clas12_flat`.

    ``process_data`` is inherited verbatim: it gates jet work on ``tr_data.jet_data``
    (empty here) and particle residuals on ``"tr_idx" in particle_data_all`` (present),
    so the dispatch lands on the vectorized overrides above.
    """

    n_workers: int = 1

    @classmethod
    def load(
        cls,
        sample: SampleT,
        config_list: list[PflowDataConfig],
        reference_label: str = "CLAS12 Recon",
        truth_label: str = "Truth",
        n_workers: int = 1,
    ) -> Self:
        data_dict: dict[str, Data] = {}
        reference_loaded = False
        for config in config_list:
            print(f"Loading data for {config.name} from {config.path}")
            raw = read_clas12_flat(
                config.path,
                entry_stop=config.num_events,
                load_reference=not reference_loaded,
                n_workers=n_workers,
            )
            event_number = raw["eventNumber"]
            if not reference_loaded:
                for source, collection, label in (
                    (Source.PF, "pflow", reference_label),
                    (Source.TR, "truth", truth_label),
                ):
                    data_dict[source.name] = Clas12FlatData.from_flat(
                        name=source.name,
                        source=source,
                        label=label,
                        collection_data=raw[collection],
                        event_number=event_number,
                    )
                reference_loaded = True
            data_dict[config.name] = Clas12FlatData.from_flat(
                name=config.name,
                source=config.pflow_type,
                label=config.label,
                collection_data=raw["fastsim"],
                event_number=event_number,
            )
        return cls(sample=sample, data_dict=data_dict, n_workers=n_workers)

    @run_once("_events_filtered", "events already filtered")
    def filter_events(self):
        numbers = [data.event_number for data in self.data_dict.values()]
        # Every collection of a single-file sample comes off the same eventNumber
        # branch, so the intersection is the identity and the shared code's
        # set-intersection plus double fancy-index is pure overhead. Rows then stay in
        # file order rather than sorted by event number; all collections stay
        # row-aligned and every downstream comparison is positional, so that is
        # equivalent.
        if all(
            first is second or np.array_equal(first, second)
            for first, second in zip(numbers, numbers[1:])
        ):
            print(
                "All collections share the same events; skipping the filter "
                f"({len(numbers[0])} events)."
            )
            for data in self.data_dict.values():
                data._events_filtered = True
            return

        good_events = functools.reduce(np.intersect1d, numbers).astype(int)
        if not good_events.size:
            raise ValueError("No common events found across all data sources.")
        for data in self.data_dict.values():
            data.filter_events(good_events)
