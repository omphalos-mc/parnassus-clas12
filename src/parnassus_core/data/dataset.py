"""Eager-loading dataset for ROOT files.

``EagerDataset`` loads all data into memory at ``__init__`` as per-collection
ragged arrays with cumsum indexing.  Collection names are read from the YAML
config ``vars_to_load`` section — nothing is hardcoded.
"""

from __future__ import annotations

import gc
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from parnassus_core.data.dataset_base import PhysicsDatasetBase
from parnassus_core.data.loaders.root_loader import ROOTFileLoader
from parnassus_core.data.physics_data import do_padding
from parnassus_core.data.utils.computed_vars import ComputedVariables
from parnassus_core.data.utils.variable_helper import CollectionInfo


def _normalize_phi(phi: np.ndarray) -> np.ndarray:
    """Wrap phi into [-pi, pi]."""
    return np.arctan2(np.sin(phi), np.cos(phi))


class EagerDataset(PhysicsDatasetBase):
    """Eager-loading dataset that stores all data in RAM.

    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    var_dict, scale_dict : str
        Paths to YAML variable-list and scaling configs.
    max_particles : int
        Padding length per collection.
    train_type : ``"particle"`` | ``"event"`` | ``"impact"``
    sin_cos : bool
        Apply sin/cos transform to phi variables.
    zero_neutral_vtx : bool
        Zero vertex-related variables for neutral particles.
    reduce_ds : float | int
        Dataset size limit (``<1.0`` fraction, ``>=1.0`` count).
    entry_start : int
        First event index.
    mode : ``"train"`` | ``"test"`` | ``"eval"``
    recalculate_ip : bool
        Recompute impact parameters from vertex positions.
    tree_name : str
        TTree name inside the ROOT file.
    num_workers : int
        Number of decompression threads for uproot.
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
        reduce_ds: float | int = 1.0,
        entry_start: int = 0,
        mode: Literal["train", "test", "eval"] = "train",
        recalculate_ip: bool = False,
        tree_name: str = "evt_tree",
        num_workers: int = 6,
    ):
        super().__init__(
            var_dict=var_dict,
            scale_dict=scale_dict,
            max_particles=max_particles,
            train_type=train_type,
            sin_cos=sin_cos,
            zero_neutral_vtx=zero_neutral_vtx,
            fs_mask_type=fs_mask_type,
            mode=mode,
            recalculate_ip=recalculate_ip,
        )

        # ---- Open loader ----
        loader = ROOTFileLoader(
            file_path,
            tree_name=tree_name,
            num_workers=num_workers,
            variable_remapping=self._vcfg.variable_remapping,
        )

        # ---- Determine event range ----
        total_events = loader.get_total_events()
        nevents = self._resolve_event_range(total_events, reduce_ds, entry_start)
        entry_stop = entry_start + nevents

        # ---- Bulk-load all branches at once ----
        all_branch_names: list[str] = []
        for coll_name, bare_vars in self._collections_to_load.items():
            all_branch_names.extend(f"{coll_name}_{v}" for v in bare_vars)
            all_branch_names.append(f"n{coll_name}")
        all_branch_names.extend(self.event_vars)

        raw_all = loader.load_branches(all_branch_names, entry_start, entry_stop)

        # ---- Split into per-collection raw data ----
        raw_collections: dict[str, dict[str, np.ndarray]] = {}
        n_particles_raw: dict[str, np.ndarray] = {}
        for coll_name, bare_vars in self._collections_to_load.items():
            raw_collections[coll_name] = {
                v: raw_all[f"{coll_name}_{v}"] for v in bare_vars
            }
            n_particles_raw[coll_name] = raw_all[f"n{coll_name}"]

        # ---- Event mask: 0 < n < max_particles for all collections ----
        event_mask = np.ones(nevents, dtype=bool)
        for coll_name in self._collections_to_load:
            n = n_particles_raw[coll_name]
            event_mask &= (n < max_particles) & (n > 0)

        # ---- PV coordinates (from truth vertices only) ----
        pv_x: np.ndarray | None = None
        pv_y: np.ndarray | None = None
        pv_z: np.ndarray | None = None
        ref_coll = list(self._collections_to_load.keys())[0]  # truth (first)

        if recalculate_ip:
            ref_data = raw_collections[ref_coll]
            pv_x, pv_y, pv_z = ComputedVariables.compute_bulk_pv_coords(
                ref_data["vx"],
                ref_data["vy"],
                ref_data["vz"],
                nevents,
            )

        # ---- Apply event mask, merge ragged → flat, preprocess ----
        self._particle_data: dict[str, dict[str, np.ndarray]] = {}

        for coll_name, raw_data in raw_collections.items():
            coll_flat: dict[str, np.ndarray] = {}
            for var, ragged in raw_data.items():
                arr = np.concatenate(ragged[event_mask])
                if "eta" in var:
                    arr = np.clip(arr, -3, 3)
                elif "phi" in var:
                    arr = _normalize_phi(arr)
                coll_flat[var] = arr

            if pv_z is not None and "vz" in raw_data:
                # vz_shifted: for ALL collections (truth + pflow)
                vz_shifted = ComputedVariables.compute_bulk_vz_shifted(
                    raw_data["vz"],
                    pv_z,
                    nevents,
                )
                coll_flat["vz_shifted"] = np.concatenate(
                    np.array(vz_shifted, dtype=object)[event_mask]
                )

                # d0My, z0My: only for non-reference (pflow) collections
                if coll_name != ref_coll and "eta" in raw_data:
                    d0_list, z0_list = ComputedVariables.compute_bulk_ip_params(
                        raw_data["eta"],
                        raw_data["phi"],
                        raw_data["vx"],
                        raw_data["vy"],
                        raw_data["vz"],
                        pv_x,  # type: ignore
                        pv_y,  # type: ignore
                        pv_z,
                        nevents,
                    )
                    coll_flat["d0My"] = np.concatenate(
                        np.array(d0_list, dtype=object)[event_mask]
                    )
                    coll_flat["z0My"] = np.concatenate(
                        np.array(z0_list, dtype=object)[event_mask]
                    )

            self._particle_data[coll_name] = coll_flat

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
            self._event_data[ev] = raw_all[ev][event_mask]

        if pv_x is not None:
            self._event_data["pv_x"] = pv_x[event_mask]
            self._event_data["pv_y"] = pv_y[event_mask]  # type: ignore
            self._event_data["pv_z"] = pv_z[event_mask]  # type: ignore

        # ---- Derived vars: ht, met, ptrel, charge (vectorized) ----
        for coll_name in self._collections_to_load:
            data = self._particle_data[coll_name]
            info = self._collection_info[coll_name]
            derived = ComputedVariables.compute_bulk_derived(
                pt_flat=data["pt"],
                phi_flat=data["phi"],
                n_particles=info.n_particles,  # type: ignore
                cumsum=info.cumsum,  # type: ignore
                pdgId_flat=data.get("pdgId"),
                class_flat=data.get("class"),
            )
            self._event_data[f"{coll_name}_ht"] = derived["ht"]
            self._event_data[f"{coll_name}_met_x"] = derived["met_x"]
            self._event_data[f"{coll_name}_met_y"] = derived["met_y"]
            data["ptrel"] = derived["ptrel"]
            if "charge" in derived:
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
                self._precomputed_means = ComputedVariables.compute_means_bulk(
                    prefixed_data,
                    primary_coll,
                    self.ctxt_vars,
                    info.n_particles,  # type: ignore
                    info.cumsum,  # type: ignore
                )

        # ---- Cleanup ----
        loader.close()
        del raw_all, raw_collections
        gc.collect()
        print("done loading data")

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
        """Extract one event, sort by ptrel desc, pad to max_particles."""
        assert isinstance(idx, int), "EagerDataset only supports int indexing"

        collection_data: dict[str, dict[str, Tensor]] = {}
        collection_masks: dict[str, Tensor] = {}

        for coll_name, info in self._collection_info.items():
            data_dict = self._particle_data[coll_name]
            start, end = info.cumsum[idx], info.cumsum[idx + 1]  # type: ignore
            n = info.n_particles[idx]  # type: ignore

            # Sort by ptrel descending
            sort_idx = np.argsort(data_dict["ptrel"][start:end])[::-1]

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
