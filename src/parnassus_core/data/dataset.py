"""Eager-loading dataset for ROOT and LUND files.

``EagerDataset`` loads all data into memory at ``__init__`` as per-collection
ragged arrays with cumsum indexing.  Collection names are read from the YAML
config ``vars_to_load`` section — nothing is hardcoded.  The file reader is
chosen from the path (see :func:`parnassus_core.data.loaders.open_file_loader`).
"""

from __future__ import annotations

import gc
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from parnassus_core.data.dataset_base import PhysicsDatasetBase
from parnassus_core.data.loaders import (
    EventSelector,
    FileLoader,
    FilteredFileLoader,
    open_file_loader,
)
from parnassus_core.data.physics_data import do_padding
from parnassus_core.data.utils.computed_vars import (
    SPECIES_COUNTS,
    ComputedVariables,
)
from parnassus_core.data.utils.variable_helper import CollectionInfo


def _normalize_phi(phi: np.ndarray) -> np.ndarray:
    """Wrap phi into [-pi, pi]."""
    return np.arctan2(np.sin(phi), np.cos(phi))


def _nonfinite_indices(arr: np.ndarray, chunk: int = 1 << 22) -> np.ndarray:
    """Indices of the NaN/inf entries of ``arr``.

    Scanned in chunks: at 15M events a full-array ``~np.isfinite(arr)`` costs a
    bool temporary the length of the whole collection, and the eager loader has
    no memory to spare.  Callers gate this on ``np.isfinite(arr.sum())``, so it
    only runs for a branch that really holds a non-finite value.
    """
    found = [
        np.flatnonzero(~np.isfinite(arr[lo : lo + chunk])) + lo
        for lo in range(0, len(arr), chunk)
    ]
    found = [f for f in found if len(f)]
    return np.concatenate(found) if found else np.empty(0, dtype=np.intp)


def _concat_kept(per_event: list[np.ndarray], keep: np.ndarray) -> np.ndarray:
    """Concatenate the per-event arrays selected by the boolean ``keep``."""
    return np.concatenate([a for a, k in zip(per_event, keep) if k])


def _compact_inplace(
    arr: np.ndarray,
    runs: list[tuple[int, int]],
    counts: np.ndarray,
    chunk: int = 1 << 22,
) -> np.ndarray:
    """Move the particles of the kept event ``runs`` to the front of ``arr``.

    ``runs`` holds ``(start, stop)`` event-index pairs to keep.  The shift is
    done in chunks and the result is a view, so no full-size copy of the branch
    is ever allocated — the dropped particles' bytes stay in the (unchanged)
    base buffer, which for a handful of events is a few tens of bytes.
    """
    dst = 0
    for ev_lo, ev_hi in runs:
        src = int(counts[:ev_lo].sum())
        remaining = int(counts[ev_lo:ev_hi].sum())
        if src == dst:  # leading run is already in place
            dst += remaining
            continue
        while remaining:
            n = min(chunk, remaining)
            arr[dst : dst + n] = arr[src : src + n].copy()
            dst += n
            src += n
            remaining -= n
    return arr[:dst]


class EagerDataset(PhysicsDatasetBase):
    """Eager-loading dataset that stores all data in RAM.

    Parameters
    ----------
    file_path : str
        Path to a ROOT file, or to LUND input (a ``.dat``/``.lund`` file, a
        directory of them, or a glob); LUND input supports eval mode only.
    var_dict, scale_dict : str
        Paths to YAML variable-list and scaling configs.
    max_particles : int
        Padding length per collection.
    train_type : ``"particle"`` | ``"event"`` | ``"impact"``
    sin_cos : bool
        Apply sin/cos transform to phi variables.
    zero_neutral_vtx : bool
        Zero vertex-related variables for neutral particles.
    fs_mask_type : ``"all"`` | ``"charged"`` | ``"neutral"``
        Target particles mask type.
    ctxt_mask_type : ``"all"`` | ``"charged"`` | ``"neutral"``
        Context particles mask type.
    reduce_ds : float | int
        Dataset size limit (``<1.0`` fraction, ``>=1.0`` count).  With an
        ``event_selector`` it counts selected events.
    entry_start : int
        First event index (of the selected events, with an ``event_selector``).
    mode : ``"train"`` | ``"test"`` | ``"eval"``
    recalculate_ip : bool
        Recompute impact parameters from vertex positions.
    tree_name : str
        TTree name inside the ROOT file.
    num_workers : int
        Number of decompression threads for uproot.
    class_based_scaling : bool
        Whether to use class-aware scaling (requires one-hot class features).
    chunk_size : int
        Number of events read from the file per chunk while loading.
        Bounds the transient memory of the ragged reads; the loaded data does
        not depend on it.
    event_selector : EventSelector or None
        Per-event selection applied while reading (see
        :class:`~parnassus_core.data.loaders.FilteredFileLoader`): only the
        selected events are loaded, so memory scales with them.
    """

    def __init__(
        self,
        file_path: str,
        var_dict: str,
        scale_dict: str,
        max_particles: int,
        train_type: Literal["particle", "event", "impact"],
        sin_cos: bool,
        zero_neutral_vtx: bool,
        fs_mask_type: Literal["all", "charged", "neutral"] = "all",
        ctxt_mask_type: Literal["all", "charged", "neutral"] = "all",
        reduce_ds: float | int = 1.0,
        entry_start: int = 0,
        mode: Literal["train", "test", "eval"] = "train",
        recalculate_ip: bool = False,
        tree_name: str = "evt_tree",
        num_workers: int = 6,
        class_based_scaling: bool = False,
        eta_clamp: tuple[float, float] | None = (-3.0, 3.0),
        chunk_size: int = 1_000_000,
        event_selector: EventSelector | None = None,
    ):
        super().__init__(
            var_dict=var_dict,
            scale_dict=scale_dict,
            max_particles=max_particles,
            train_type=train_type,
            sin_cos=sin_cos,
            zero_neutral_vtx=zero_neutral_vtx,
            fs_mask_type=fs_mask_type,
            ctxt_mask_type=ctxt_mask_type,
            mode=mode,
            recalculate_ip=recalculate_ip,
            class_based_scaling=class_based_scaling,
            eta_clamp=eta_clamp,
        )

        # ---- Open loader ----
        # Events past this index are never read, so readers that parse
        # sequentially (LUND) can stop there.
        event_cap = entry_start + int(reduce_ds) if reduce_ds >= 1.0 else None
        loader: FileLoader = open_file_loader(
            file_path,
            tree_name=tree_name,
            num_workers=num_workers,
            variable_remapping=self._vcfg.variable_remapping,
            # With a selector the cap applies to selected events, not file events.
            max_events=None if event_selector is not None else event_cap,
        )
        if event_selector is not None:
            loader = FilteredFileLoader(loader, event_selector, max_kept=event_cap)

        # ---- Determine event range ----
        total_events = loader.get_total_events()
        nevents = self._resolve_event_range(total_events, reduce_ds, entry_start)
        entry_stop = entry_start + nevents

        # ---- Scalar branches: particle counts + event-level variables ----
        # These are small enough to read for the whole range at once; the
        # ragged particle branches are streamed in chunks further down.
        flat_branch_names = [f"n{c}" for c in self._collections_to_load]
        flat_branch_names += self.event_vars
        flat_raw = loader.load_branches(
            list(dict.fromkeys(flat_branch_names)), entry_start, entry_stop
        )
        n_particles_raw: dict[str, np.ndarray] = {
            c: flat_raw[f"n{c}"] for c in self._collections_to_load
        }

        ref_coll = list(self._collections_to_load.keys())[0]  # truth (first)

        # ---- Event mask: n < max_particles for all collections ----

        allow_empty = self.train_type == "event"
        event_mask = np.ones(nevents, dtype=bool)
        for coll_name in self._collections_to_load:
            n = n_particles_raw[coll_name]
            event_mask &= n < max_particles
            if not (allow_empty and coll_name != ref_coll):
                event_mask &= n > 0

        # ---- PV coordinates (from truth vertices only), filled per chunk ----
        pv_x: np.ndarray | None = None
        pv_y: np.ndarray | None = None
        pv_z: np.ndarray | None = None

        if recalculate_ip:
            pv_x = np.empty(nevents, dtype=np.float32)
            pv_y = np.empty(nevents, dtype=np.float32)
            pv_z = np.empty(nevents, dtype=np.float32)

        # ---- Stream ragged branches: apply event mask, flatten, preprocess ----
        # uproot's ``library="np"`` output holds one numpy view object per event
        # per branch, several times the size of the values themselves.  Reading
        # it chunk by chunk into flat arrays sized up front from the counts
        # keeps the peak memory close to the final in-memory footprint.
        kept_cumsum: dict[str, np.ndarray] = {
            c: np.concatenate([[0], np.cumsum(n[event_mask])])
            for c, n in n_particles_raw.items()
        }
        self._particle_data: dict[str, dict[str, np.ndarray]] = {
            c: {} for c in self._collections_to_load
        }

        # Events carrying a NaN/inf are dropped: the value survives every
        # downstream transform, reaches the loss, and the global-norm gradient
        # clip then turns it into a NaN gradient for *every* parameter.  Flags
        # index the events kept by the count mask above; collected across all
        # collections, applied once the loop below is done.
        drop_events = np.zeros(int(event_mask.sum()), dtype=bool)

        particle_branch_names = [
            f"{c}_{v}"
            for c, bare_vars in self._collections_to_load.items()
            for v in bare_vars
        ]
        ev_lo = 0  # first event of the chunk, relative to entry_start
        kept_lo = 0  # number of kept events before the chunk
        for chunk in loader.iter_branches(
            particle_branch_names, entry_start, entry_stop, chunk_size
        ):
            ev_hi = ev_lo + len(chunk[particle_branch_names[0]])
            chunk_mask = event_mask[ev_lo:ev_hi]
            kept_hi = kept_lo + int(chunk_mask.sum())

            chunk_pv = None
            if pv_x is not None and pv_y is not None and pv_z is not None:
                pv_x[ev_lo:ev_hi], pv_y[ev_lo:ev_hi], pv_z[ev_lo:ev_hi] = (
                    ComputedVariables.compute_ragged_pv_coords(
                        chunk[f"{ref_coll}_vx"],
                        chunk[f"{ref_coll}_vy"],
                        chunk[f"{ref_coll}_vz"],
                    )
                )
                chunk_pv = (pv_x[ev_lo:ev_hi], pv_y[ev_lo:ev_hi], pv_z[ev_lo:ev_hi])

            if kept_hi > kept_lo:
                for coll_name, bare_vars in self._collections_to_load.items():
                    flat, nonfinite = self._flatten_chunk(
                        {v: chunk[f"{coll_name}_{v}"] for v in bare_vars},
                        chunk_mask,
                        chunk_pv,
                        is_ref=coll_name == ref_coll,
                    )
                    cumsum = kept_cumsum[coll_name]
                    p_lo, p_hi = cumsum[kept_lo], cumsum[kept_hi]
                    coll_flat = self._particle_data[coll_name]
                    for var, arr in flat.items():
                        if var not in coll_flat:
                            coll_flat[var] = np.empty(int(cumsum[-1]), dtype=arr.dtype)
                        coll_flat[var][p_lo:p_hi] = arr
                    if len(nonfinite):
                        bad_events = (
                            np.searchsorted(cumsum, p_lo + nonfinite, side="right") - 1
                        )
                        drop_events[bad_events] = True

            del chunk
            ev_lo, kept_lo = ev_hi, kept_hi

        # ---- Drop events holding non-finite values ----
        if drop_events.any():
            kept_indices = np.nonzero(event_mask)[0]
            file_indices = (entry_start + kept_indices[drop_events])[:10].tolist()
            print(
                f"WARNING: dropping {int(drop_events.sum())} event(s) with "
                f"non-finite values; file indices {file_indices}"
                f"{' ...' if int(drop_events.sum()) > 10 else ''}"
            )
            # Maximal runs of kept events; the particles of each run are shifted
            # down onto the dropped ones in place (see _compact_inplace).
            edges = np.diff(
                np.concatenate([[0], (~drop_events).view(np.int8), [0]]).astype(np.int8)
            )
            runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
            for coll_name, coll_flat in self._particle_data.items():
                counts = n_particles_raw[coll_name][event_mask]
                for var, arr in coll_flat.items():
                    coll_flat[var] = _compact_inplace(arr, runs, counts)
            event_mask[kept_indices[drop_events]] = False

        # ---- Particle counts + cumsum ----
        self._collection_info: dict[str, CollectionInfo] = {}
        for coll_name in self._collections_to_load:
            n_masked = n_particles_raw[coll_name][event_mask]
            cumsum = np.cumsum(np.concatenate([[0], n_masked]))
            self._collection_info[coll_name] = CollectionInfo(
                name=coll_name,
                n_particles=n_masked,
                cumsum=cumsum,
            )

        self._nevents = int(event_mask.sum())
        print(f" we have  {self._nevents}  events")

        # ---- Event-level data (from file) ----
        self._event_data: dict[str, np.ndarray] = {}
        for ev in self.event_vars:
            self._event_data[ev] = flat_raw[ev][event_mask]

        if pv_x is not None:
            self._event_data["pv_x"] = pv_x[event_mask]
            self._event_data["pv_y"] = pv_y[event_mask]  # type: ignore
            self._event_data["pv_z"] = pv_z[event_mask]  # type: ignore

        # ---- Derived vars: ht_p, species counts, Pxh/Pyh, charge (vectorized) ----
        for coll_name in self._collections_to_load:
            data = self._particle_data[coll_name]
            info = self._collection_info[coll_name]

            missing_kinematics = {"p", "theta", "phi"} - data.keys()
            if missing_kinematics:
                raise ValueError(
                    f"Collection '{coll_name}' needs {sorted(missing_kinematics)} "
                    "in vars_to_load: the derived event variables and the "
                    "particle ordering are built from p, theta and phi."
                )

            # Ng/Nn and Pxh/Pyh are the pdgId-derived event vars. Without pdgId
            # they would just be absent, surfacing much later as a KeyError from
            # build_fs_data / build_ctxt_global_data, so fail here with the
            # actual cause.
            if "pdgId" not in data:
                missing = (SPECIES_COUNTS.keys() | {"Pxh", "Pyh"}) & {
                    self.var_helper.get_base_name(v)
                    for v in [*self.fs_vars, *self.ctxt_global_vars]
                    if self.var_helper.get_collection(v) == coll_name
                }
                if missing:
                    raise ValueError(
                        f"Cannot compute {sorted(missing)} for collection "
                        f"'{coll_name}': add '{coll_name}_pdgId' to vars_to_load."
                    )

            derived = ComputedVariables.compute_flat_derived(
                p_flat=data["p"],
                theta_flat=data["theta"],
                phi_flat=data["phi"],
                n_particles=info.n_particles,  # type: ignore
                cumsum=info.cumsum,  # type: ignore
                pdgId_flat=data.get("pdgId"),
                class_flat=data.get("class"),
            )
            # Everything compute_flat_derived returns is event-level except
            # charge, which is per-particle and handled just below.
            for key, values in derived.items():
                if key != "charge":
                    self._event_data[f"{coll_name}_{key}"] = values
            # Prefer a stored charge branch; only fall back to the pdgId-derived
            # charge when none was loaded.
            if "charge" in derived and "charge" not in data:
                data["charge"] = derived["charge"]

        # ---- d0Sig / z0Sig ----
        for coll_name, data in self._particle_data.items():
            if "d0" in data and "d0Error" in data:
                data["d0Sig"], data["z0Sig"] = ComputedVariables.compute_significance(  # type: ignore
                    data["d0"],
                    data["d0Error"],
                    data["z0"],
                    data["z0Error"],
                )

        # ---- Pre-compute means ----
        self._precomputed_means: np.ndarray | None = None
        if "means" in self.ctxt_global_vars:
            primary_coll = self.var_helper.get_collection(self.ctxt_vars[0])
            if primary_coll is not None and primary_coll in self._collection_info:
                info = self._collection_info[primary_coll]
                prefixed_data = {
                    f"{primary_coll}_{k}": v
                    for k, v in self._particle_data[primary_coll].items()
                }
                self._precomputed_means = ComputedVariables.compute_flat_means(
                    prefixed_data,
                    primary_coll,
                    self.ctxt_vars,
                    info.n_particles,  # type: ignore
                    info.cumsum,  # type: ignore
                )

        # ---- Cleanup ----
        loader.close()
        del flat_raw
        gc.collect()
        print("done loading data")

    def _flatten_chunk(
        self,
        raw_data: dict[str, np.ndarray],
        chunk_mask: np.ndarray,
        pv: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        is_ref: bool,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Flatten one chunk of a collection's ragged branches.

        Parameters
        ----------
        raw_data : dict[str, np.ndarray]
            ``{bare_var: ragged_array}`` for every event of the chunk.
        chunk_mask : np.ndarray
            Boolean mask of the chunk's events to keep.
        pv : tuple of np.ndarray or None
            ``(pv_x, pv_y, pv_z)`` for every event of the chunk; when given,
            ``vz_shifted`` is derived, plus ``d0My``/``z0My`` unless ``is_ref``.
        is_ref : bool
            Whether this is the reference (truth) collection.

        Returns
        -------
        flat : dict[str, np.ndarray]
            Flat per-particle arrays of the kept events; floating-point
            branches are stored as float32.
        nonfinite : np.ndarray
            Indices into ``flat`` of particles with a NaN/inf in a loaded
            branch, checked before any preprocessing (clipping would hide an
            inf).  May contain duplicates.
        """
        flat: dict[str, np.ndarray] = {}
        nonfinite: list[np.ndarray] = []
        for var, ragged in raw_data.items():
            arr = np.concatenate(ragged[chunk_mask])
            # __getitem__ hands these to the model as float32 anyway; keeping
            # the ROOT float64 doubles the resident dataset (one copy per DDP
            # rank) for no gain.  Cast before the finiteness check so a value
            # that overflows float32 is caught there.
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32)
            # A sum cannot hide a non-finite value (NaN propagates, and
            # inf + -inf is NaN), so this gate is an allocation-free way to
            # keep the element-wise scan off the common, clean path.
            if arr.dtype.kind == "f" and not np.isfinite(arr.sum()):
                nonfinite.append(_nonfinite_indices(arr))
            # Exact base-name tests: "eta" in "theta" is True, and theta must
            # not be clamped to the eta acceptance.
            if var == "eta":
                if self.eta_clamp is not None:
                    arr = np.clip(arr, self.eta_clamp[0], self.eta_clamp[1])
            elif var == "phi":
                arr = _normalize_phi(arr)
            flat[var] = arr

        if pv is not None and "vz" in raw_data:
            pv_x, pv_y, pv_z = pv
            # vz_shifted: for ALL collections (truth + pflow)
            vz_shifted = ComputedVariables.compute_ragged_vz_shifted(
                raw_data["vz"], pv_z
            )
            flat["vz_shifted"] = _concat_kept(vz_shifted, chunk_mask).astype(np.float32)

            # d0My, z0My: only for non-reference (pflow) collections
            if not is_ref and "eta" in raw_data:
                d0_list, z0_list = ComputedVariables.compute_ragged_ip_params(
                    raw_data["eta"],
                    raw_data["phi"],
                    raw_data["vx"],
                    raw_data["vy"],
                    raw_data["vz"],
                    pv_x,
                    pv_y,
                    pv_z,
                )
                flat["d0My"] = _concat_kept(d0_list, chunk_mask).astype(np.float32)
                flat["z0My"] = _concat_kept(z0_list, chunk_mask).astype(np.float32)

        if not nonfinite:
            return flat, np.empty(0, dtype=np.intp)
        return flat, np.concatenate(nonfinite)

    # ------------------------------------------------------------------
    # PhysicsDatasetBase interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._nevents

    def _get_collections(
        self,
        idx: int | slice,
    ) -> tuple[
        dict[str, dict[str, Tensor]],
        dict[str, Tensor],
        dict[str, Tensor],
        Tensor | None,
    ]:
        """Extract one event, sort by p descending, pad to max_particles."""
        assert isinstance(idx, int), "EagerDataset only supports int indexing"

        collection_data: dict[str, dict[str, Tensor]] = {}
        collection_masks: dict[str, Tensor] = {}

        for coll_name, info in self._collection_info.items():
            data_dict = self._particle_data[coll_name]
            start, end = info.cumsum[idx], info.cumsum[idx + 1]  # type: ignore
            n = info.n_particles[idx]  # type: ignore

            # Sort by p descending
            sort_idx = np.argsort(data_dict["p"][start:end])[::-1]

            coll: dict[str, Tensor] = {}
            for var, arr in data_dict.items():
                t = torch.tensor(
                    arr[start:end][sort_idx].copy(),
                    dtype=torch.float32,
                )
                coll[var] = do_padding(t, self.max_particles)
            collection_data[coll_name] = coll

            mask = torch.zeros(self.max_particles, dtype=torch.bool)
            mask[:n] = True
            collection_masks[coll_name] = mask

        # Event-level data
        event_data: dict[str, Tensor] = {}
        for k, v in self._event_data.items():
            val = v[idx]
            if v.dtype == np.uint64:
                event_data[k] = torch.tensor(int(val), dtype=torch.int64)
            else:
                event_data[k] = torch.tensor(val)

        # pv_coords as a single tensor for the base class to forward
        if self.recalculate_ip and "pv_x" in event_data:
            event_data["pv_coords"] = torch.stack(
                [event_data["pv_x"], event_data["pv_y"], event_data["pv_z"]]
            ).float()

        # Pre-computed means
        means: Tensor | None = None
        if self._precomputed_means is not None:
            means = torch.tensor(
                self._precomputed_means[idx],
                dtype=torch.float32,
            )

        return collection_data, collection_masks, event_data, means
