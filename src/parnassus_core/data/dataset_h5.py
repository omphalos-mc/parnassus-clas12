"""Streaming dataset for HDF5 files.

``StreamingDataset`` reads batches on-the-fly from an HDF5 file produced
by ``root_to_hdf5.py``.  Collection names are read from the YAML config
``vars_to_load`` section — nothing is hardcoded.

Use with :class:`RandomBatchSampler` so that ``__getitem__`` receives a
``slice`` rather than an ``int``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

from parnassus_core.data.utils.computed_vars import (
    ComputedVariables,
    calculate_impact_params,
    get_pv_coords,
)
from parnassus_core.data.dataset_base import PhysicsDatasetBase
from parnassus_core.data.loaders.hdf5_loader import HDF5FileLoader


class StreamingDataset(PhysicsDatasetBase):
    """Streaming dataset that reads batches from an HDF5 file.

    Data is **not** loaded at init — only at ``__getitem__`` time.  The
    HDF5 file handle is kept open for the dataset's lifetime.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    var_dict, scale_dict : str
        Paths to YAML variable-list and scaling configs.
    max_particles : int
        Padding length per collection (must match the HDF5 padding dim).
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
    class_based_scaling : bool
        Whether to use class-aware scaling (requires one-hot class features).
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
        class_based_scaling: bool = False,
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
        )
        self.entry_start = entry_start

        # ---- Open loader ----
        self.loader = HDF5FileLoader(
            file_path,
            variable_remapping=self._vcfg.variable_remapping,
        )

        # ---- Determine event range ----
        total = self.loader.get_total_events()
        self._nevents = self._resolve_event_range(total, reduce_ds, entry_start)

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
        """Read one batch from HDF5 and build the common intermediate format.

        For single-event access (``idx`` is ``int``), wraps in a
        1-element slice and squeezes the batch dim so the output shape
        is ``(N,)`` / ``(N, F)`` — matching the eager dataset contract.
        """
        squeeze = False
        if isinstance(idx, int):
            start = self.entry_start + idx
            batch_idx = slice(start, start + 1)
            squeeze = True
        else:
            batch_idx = slice(
                self.entry_start + idx.start,
                self.entry_start + idx.stop,
            )

        # ---- Read event-level data ----
        event_raw = self.loader.read_events(batch_idx)
        event_data: dict[str, Tensor] = {}
        for k, v in event_raw.items():
            if k == "eventNumber":
                # Preserve integer type for eventNumber (HDF5 may store as
                # float in the structured array, but downstream expects int64).
                event_data[k] = torch.from_numpy(v).long()
            else:
                event_data[k] = torch.from_numpy(v).float()

        # ---- Read collections ----
        collection_data: dict[str, dict[str, Tensor]] = {}
        collection_masks: dict[str, Tensor] = {}
        ref_coll = self._all_collection_names[0]  # truth

        for coll_name in self._collections_to_load:
            raw, valid = self.loader.read_collection(batch_idx, coll_name)

            data: dict[str, Tensor] = {}
            for field, arr in raw.items():
                t = torch.from_numpy(arr).float()
                t = t.nan_to_num(0.0)  # HDF5 padding has NaN
                data[field] = t

            mask = torch.from_numpy(valid).bool()

            # ---- Preprocessing (matching EagerDataset) ----
            if "eta" in data:
                data["eta"] = data["eta"].clamp(-3.0, 3.0)
            if "phi" in data:
                data["phi"] = torch.atan2(
                    torch.sin(data["phi"]),
                    torch.cos(data["phi"]),
                )

            # ---- Derived: ptrel ----
            ht_key = f"{coll_name}_ht"
            if "pt" in data and ht_key in event_data:
                ht = event_data[ht_key]  # (B,) or scalar
                if ht.dim() == 0:
                    ht = ht.unsqueeze(0)
                data["ptrel"] = data["pt"] / ht.unsqueeze(-1)

            # ---- Derived: charge ----
            # New HDF5 format has charge as a stored field; only compute
            # from pdgId + class if the file did not provide it.
            if "charge" not in data:
                if "pdgId" in data and "class" in data:
                    charge = torch.sign(data["pdgId"])
                    charge[data["class"] > 2] = 0.0
                    data["charge"] = charge

            collection_data[coll_name] = data
            collection_masks[coll_name] = mask

        # ---- PV coordinates ----
        # New HDF5 format stores Sim_PV_x/y/z in the events dataset.
        # Fall back to majority-vote from truth vertices (old format).
        has_stored_pv = all(k in event_data for k in ("pv_x", "pv_y", "pv_z"))

        if has_stored_pv:
            # ---- New format: PV stored in events ----
            pv_x_t = event_data["pv_x"]
            pv_y_t = event_data["pv_y"]
            pv_z_t = event_data["pv_z"]

            # ---- pv_coords as single tensor ----
            event_data["pv_coords"] = torch.stack(
                [pv_x_t, pv_y_t, pv_z_t],
                dim=-1,
            )

        elif ref_coll in collection_data:
            # ---- Old format: majority vote from truth vertices ----
            truth = collection_data[ref_coll]
            valid_np = collection_masks[ref_coll].numpy()
            B = valid_np.shape[0]

            vx_np = truth["vx"].numpy()
            vy_np = truth["vy"].numpy()
            vz_np = truth["vz"].numpy()

            pv_x_arr = np.empty(B, dtype=np.float32)
            pv_y_arr = np.empty(B, dtype=np.float32)
            pv_z_arr = np.empty(B, dtype=np.float32)

            for i in range(B):
                m = valid_np[i]
                pv_x_arr[i] = get_pv_coords(vx_np[i][m])
                pv_y_arr[i] = get_pv_coords(vy_np[i][m])
                pv_z_arr[i] = get_pv_coords(vz_np[i][m])

            pv_x_t = torch.from_numpy(pv_x_arr)
            pv_y_t = torch.from_numpy(pv_y_arr)
            pv_z_t = torch.from_numpy(pv_z_arr)

            event_data["pv_x"] = pv_x_t
            event_data["pv_y"] = pv_y_t
            event_data["pv_z"] = pv_z_t

            # ---- pv_coords as single tensor ----
            event_data["pv_coords"] = torch.stack(
                [pv_x_t, pv_y_t, pv_z_t],
                dim=-1,
            )
        else:
            pv_x_t = pv_y_t = pv_z_t = None

        if pv_x_t is not None:
            assert pv_y_t is not None and pv_z_t is not None

        if self.recalculate_ip:
            if pv_x_t is not None:
                assert pv_y_t is not None and pv_z_t is not None

                # ---- vz_shifted for all collections ----
                for cn, cdata in collection_data.items():
                    if "vz" in cdata:
                        cdata["vz_shifted"] = cdata["vz"] - pv_z_t.unsqueeze(-1)

                # ---- d0My / z0My for non-reference collections only ----
                pv_x_np = pv_x_t.numpy().reshape(-1, 1)
                pv_y_np = pv_y_t.numpy().reshape(-1, 1)
                pv_z_np = pv_z_t.numpy().reshape(-1, 1)
                for cn, cdata in collection_data.items():
                    if cn == ref_coll:
                        continue
                    if "eta" not in cdata or "vx" not in cdata:
                        continue
                    d0, z0 = calculate_impact_params(
                        cdata["eta"].numpy(),
                        cdata["phi"].numpy(),
                        cdata["vx"].numpy(),
                        cdata["vy"].numpy(),
                        cdata["vz"].numpy(),
                        pv_x_np,
                        pv_y_np,
                        pv_z_np,
                    )
                    cdata["d0My"] = torch.from_numpy(d0).float()
                    cdata["z0My"] = torch.from_numpy(z0).float()

        # ---- d0Sig / z0Sig for collections that have d0/d0Error ----
        for cdata in collection_data.values():
            if "d0" in cdata and "d0Error" in cdata:
                d0s, z0s = ComputedVariables.compute_significance(
                    cdata["d0"],
                    cdata["d0Error"],
                    cdata["z0"],
                    cdata["z0Error"],
                )
                cdata["d0Sig"] = d0s  # type: ignore
                cdata["z0Sig"] = z0s  # type: ignore

        # ---- Pad collections to self.max_particles if HDF5 dim is smaller ----
        for cn in collection_data:
            pad = self.max_particles - collection_masks[cn].shape[-1]
            if pad > 0:
                collection_masks[cn] = torch.nn.functional.pad(
                    collection_masks[cn], (0, pad), value=False
                )
                for field, t in collection_data[cn].items():
                    collection_data[cn][field] = torch.nn.functional.pad(t, (0, pad))

        # ---- Drop events with 0 valid particles in any collection ----
        if not squeeze:
            keep = None
            for mask in collection_masks.values():
                has_particles = mask.any(dim=-1)  # (B,)
                keep = has_particles if keep is None else keep & has_particles
            if keep is not None and not keep.all():
                print(
                    f"Warning: dropping {len(keep) - keep.sum()} events with no valid particles"
                )
                for cn in collection_data:
                    collection_data[cn] = {
                        k: v[keep] for k, v in collection_data[cn].items()
                    }
                    collection_masks[cn] = collection_masks[cn][keep]
                event_data = {k: v[keep] for k, v in event_data.items()}

        # ---- Squeeze batch dim for single-event access ----
        if squeeze:
            for cn in collection_data:
                collection_data[cn] = {
                    k: v.squeeze(0) for k, v in collection_data[cn].items()
                }
                collection_masks[cn] = collection_masks[cn].squeeze(0)
            event_data = {
                k: v.squeeze(0) if v.dim() >= 1 and v.shape[0] == 1 else v
                for k, v in event_data.items()
            }

        # means=None → computed per-batch in __getitem__
        return collection_data, collection_masks, event_data, None
