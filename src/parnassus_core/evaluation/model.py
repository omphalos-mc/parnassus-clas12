from typing import cast


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.modules.attention import Attention
from parnassus_core.data.utils.computed_vars import SPECIES_COUNTS
from parnassus_core.data.utils.scaling import FeatureScaler, Unscaler

from parnassus_core.evaluation.config import EvaluationConfig, ModelConfig
from parnassus_core.data.utils.variable_helper import VariableHelper
from parnassus_core.utils.pid import N_PID_CLASSES

VarList = list[str]


class EvaluationModel(nn.Module):
    def __init__(self, config: EvaluationConfig):
        super().__init__()

        self.recalculate_ip = config.recalculate_ip

        self.event_model = self.load_lightning_model(config.event_model)
        self.particle_model = self.load_lightning_model(config.particle_model)
        if config.impact_model is not None:
            self.impact_model = self.load_lightning_model(config.impact_model)
        else:
            self.impact_model = None

        self.torch_compile = config.torch_compile
        self.trim_padding = config.trim_padding
        if self.trim_padding:
            # Plain flash attention takes no padding mask, so padded slots reach
            # the real ones there and trimming them would change the output.
            unmasked = [
                name
                for model in self._flow_models()
                for name, module in model.named_modules()
                if isinstance(module, Attention) and module.attn_type == "flash"
            ]
            if unmasked:
                raise ValueError(
                    "trim_padding requires masked attention, but these modules "
                    f"use unmasked 'flash': {unmasked}"
                )

        self.feature_scaler = FeatureScaler(
            config.scale_dict,
            class_based_scaling=config.data_config.class_based_scaling,
        )

        self.fs_vars = config.var_config.fs_vars
        self.ctxt_vars: VarList = config.var_config.ctxt_vars
        self.ctxt_global_vars: VarList = (
            config.var_config.ctxt_global_vars + self.fs_vars["event"]
        )

        self.variable_helper = VariableHelper(collection_names=["truth", "pflow"])

        self.max_particles = config.data_config.max_particles
        self.sin_cos = True

        self.unscaler = Unscaler(
            feature_scaler=self.feature_scaler,
            ctxt_vars=self.ctxt_vars,
            fs_vars=self.fs_vars["particle"],
            ctxt_global_vars=self.ctxt_global_vars,
            sin_cos=self.sin_cos,
        )

        self.fs_var_idx_dict = self._get_fs_var_idxs()
        self.ht_shift = self.feature_scaler[self.ht_key].shift
        self.ht_scale = self.feature_scaler[self.ht_key].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale

    def _get_fs_var_idxs(self) -> dict[str, int]:
        idx_dict = {}
        # Event models trained on the p analogue of ht carry ht_p instead; the
        # empty-event veto in generate_event then uses the ht_p scaler.
        coll = self.unscaler.target_collection
        event_vars = self.fs_vars["event"]
        self.ht_key = (
            "ht_p"
            if f"{coll}_ht" not in event_vars and f"{coll}_ht_p" in event_vars
            else "ht"
        )
        idx_dict["ht"] = event_vars.index(f"{coll}_{self.ht_key}")
        idx_dict["npart"] = self.fs_vars["event"].index(
            f"n{self.unscaler.target_collection}"
        )
        # Species counts (Ng, Nn) are optional event targets, rounded like npart.
        for name in SPECIES_COUNTS:
            var = f"{self.unscaler.target_collection}_{name}"
            if var in self.fs_vars["event"]:
                idx_dict[name] = self.fs_vars["event"].index(var)
        # Build a dict mapping variable names to their index in the particle feature vector.
        # Per-variable encoded widths must mirror data/utils/transforms.py:
        #   phi -> sin/cos (2), charge -> one-hot (3), class -> one-hot (5),
        #   pdgId -> one-hot (N_PID_CLASSES), everything else -> 1.
        curr_idx = 0
        for var in self.fs_vars["particle"]:
            var = self.variable_helper.scaler_key(var)
            if var in {"vx", "vy", "vz", "vz_shifted"}:
                idx_dict[var] = curr_idx
            if var == "phi":
                if self.sin_cos:
                    idx_dict["sin_phi"] = curr_idx
                    idx_dict["cos_phi"] = curr_idx + 1
                    curr_idx += 2
                else:
                    idx_dict["phi"] = curr_idx
                    curr_idx += 1
            elif var == "charge":
                idx_dict[var] = curr_idx
                curr_idx += 3
            elif var == "class":
                idx_dict[var] = curr_idx
                curr_idx += 5
            elif var == "pdgId":
                idx_dict[var] = curr_idx
                curr_idx += N_PID_CLASSES
            else:
                idx_dict[var] = curr_idx
                curr_idx += 1

        # Total encoded particle-feature width (used to shape the particle sample).
        self.fs_part_dim = curr_idx

        return idx_dict

    def _flow_models(self) -> list[FlowLightning]:
        models = [self.event_model, self.particle_model, self.impact_model]
        return [model for model in models if model is not None]

    def compile_models(self):
        """Compile the networks the samplers call, if ``torch_compile`` is set.

        Compiling the ``FlowLightning`` wrapper would only cover its ``forward``
        (the training loss): ``sample()`` reaches the network through the
        sampler's ``model_fn``, which looks up ``self.net`` on every call.
        ``dynamic=True`` because the batch size changes with the empty-event
        veto, and the trimmed particle lengths change from batch to batch.
        """
        if not self.torch_compile:
            return
        for model in self._flow_models():
            model.net = torch.compile(model.net, dynamic=True)

    def _trimmed(
        self, ctxt_dict: dict[str, torch.Tensor], fs_len: int | None = None
    ) -> dict[str, torch.Tensor]:
        """``ctxt_dict`` without the particle slots no event in the batch uses.

        Exact: the networks only mix particles through masked attention and
        masked pooling, so a padded slot never reaches a real one (checked for
        unmasked attention in ``__init__``). Returns a shallow copy, leaving
        ``ctxt_dict`` at full width for unscaling; ``fs_mask`` is cut to
        ``fs_len`` when given. Returns ``ctxt_dict`` itself when
        ``trim_padding`` is off.
        """
        if not self.trim_padding:
            return ctxt_dict
        ctxt_mask = ctxt_dict["ctxt_mask"]
        # One past the last slot any event uses, so no real particle is cut
        # even if the mask were not a prefix.
        used = ctxt_mask.any(0).nonzero()
        if not used.numel():
            return ctxt_dict
        ctxt_len = int(used.max()) + 1
        trimmed = ctxt_dict | {
            "ctxt_data": ctxt_dict["ctxt_data"][:, :ctxt_len],
            "ctxt_mask": ctxt_mask[:, :ctxt_len],
        }
        if fs_len is not None:
            trimmed["fs_mask"] = ctxt_dict["fs_mask"][:, :fs_len]
        return trimmed

    def load_lightning_model(self, model_config: ModelConfig) -> FlowLightning:
        model = FlowLightning.load_from_checkpoint(
            model_config.ckpt_path,
            map_location=torch.device("cpu"),
            weights_only=False,  # bypass security for now and load the full checkpoint
        ).eval()
        model.change_sampler(
            model_config.sampler_type,
            model_config.n_steps,
            **model_config.sampler_kwargs,
        )
        return model

    def unscale_impact(self, fs_data: torch.Tensor) -> dict[str, np.ndarray]:
        fs_data = fs_data.cpu()
        new_data_dict = {}
        var_idx = 0
        for var_name in self.fs_vars["impact"]:
            var_name = self.variable_helper.scaler_key(var_name)
            fs_, shift = self.unscaler.extract_var(
                fs_data, var_name, var_idx, self.sin_cos
            )
            var_idx += shift
            new_data_dict[f"fs_{var_name}"] = fs_.numpy()
        return new_data_dict

    def generate_event(self, ctxt_dict):
        ctxt_dict["fs_mask"] = None
        event_shape = (
            ctxt_dict["ctxt_global_data"].shape[0],
            len(self.fs_vars["event"]),
        )
        fs_evt = cast(
            torch.Tensor,
            self.event_model.sample(
                event_shape,
                self._trimmed(ctxt_dict),
                to_cpu=False,
            ),
        )
        fs_ht = fs_evt[..., self.fs_var_idx_dict["ht"]]
        # The npart target is an exact integer during training
        # (``collection_masks[coll].sum(-1)``, see data/physics_data.py), so the
        # matching inverse is the nearest integer.  Rounding also makes "the
        # event model predicted no reconstructed particles" an exact test — the
        # event stage trains on npflow == 0 events, so its output has real mass
        # near zero.
        fs_npart = torch.round(
            self.feature_scaler["npart"].inverse_transform(
                fs_evt[..., self.fs_var_idx_dict["npart"]]
            )
        )
        # Events the event model reconstructed nothing in are skipped: the
        # particle stage has no targets to generate for them.
        good_evt_mask = (
            (fs_npart >= 1)
            & (fs_ht > self.min_ht_scaled)
            & (fs_npart < self.max_particles)
        )
        bad_idxs = torch.argwhere(~good_evt_mask).flatten().cpu().numpy()
        for key, value in ctxt_dict.items():
            if isinstance(value, torch.Tensor):
                ctxt_dict[key] = value[good_evt_mask]
        fs_npart = fs_npart[good_evt_mask]
        fs_evt = fs_evt[good_evt_mask]
        fs_mask = torch.arange(self.max_particles, device=fs_evt.device).expand(
            fs_evt.shape[0], self.max_particles
        ) < fs_npart.unsqueeze(1)
        # Re-encode the rounded count so the multiplicity the particle stage is
        # conditioned on equals the number of slots in fs_mask, as in training.
        fs_evt[..., self.fs_var_idx_dict["npart"]] = self.feature_scaler[
            "npart"
        ].transform(fs_npart)
        # Species counts are exact integers in training too; round and re-encode
        # them the same way. Zero is a valid count, so clamp instead of filtering.
        for name in SPECIES_COUNTS.keys() & self.fs_var_idx_dict.keys():
            idx = self.fs_var_idx_dict[name]
            count = torch.round(
                self.feature_scaler[name].inverse_transform(fs_evt[..., idx])
            ).clamp(min=0)
            fs_evt[..., idx] = self.feature_scaler[name].transform(count)
        ctxt_dict["ctxt_global_data"] = torch.cat(
            [ctxt_dict["ctxt_global_data"], fs_evt], -1
        )
        ctxt_dict["fs_mask"] = fs_mask
        fs_part_shape = (fs_evt.shape[0], self.max_particles, self.fs_part_dim)
        # With trim_padding, sample only up to the largest generated
        # multiplicity in the batch, then pad back so everything downstream
        # keeps the max_particles layout.
        fs_len = self.max_particles
        if self.trim_padding and fs_npart.numel():
            fs_len = int(fs_npart.max())
        fs_part = cast(
            torch.Tensor,
            self.particle_model.sample(
                (fs_part_shape[0], fs_len, self.fs_part_dim),
                self._trimmed(ctxt_dict, fs_len),
                to_cpu=False,
            ),
        )
        if fs_len < self.max_particles:
            fs_part = F.pad(fs_part, (0, 0, 0, self.max_particles - fs_len))
        if self.impact_model is not None:
            fs_kin_feats = fs_part[..., : self.fs_var_idx_dict["class"]]
            fs_class_raw = fs_part[..., self.fs_var_idx_dict["class"] :]
            ctxt_dict["pf_ctxt_data"] = fs_part
            if self.recalculate_ip:
                fs_mask = ctxt_dict["fs_mask"]

                vx = self.feature_scaler["vx"].inverse_transform(
                    fs_kin_feats[..., self.fs_var_idx_dict["vx"]]
                )
                vy = self.feature_scaler["vy"].inverse_transform(
                    fs_kin_feats[..., self.fs_var_idx_dict["vy"]]
                )
                if "vz_shifted" in self.fs_var_idx_dict:
                    fs_kin_vz_shifted_raw = fs_kin_feats[
                        ..., self.fs_var_idx_dict["vz_shifted"]
                    ]
                    fs_kin_vz_shifted = self.feature_scaler[
                        "vz_shifted"
                    ].inverse_transform(fs_kin_vz_shifted_raw)
                elif "vz" in self.fs_var_idx_dict:
                    fs_kin_vz_raw = fs_kin_feats[..., self.fs_var_idx_dict["vz"]]
                    fs_kin_vz = self.feature_scaler["vz"].inverse_transform(
                        fs_kin_vz_raw
                    )
                    fs_kin_vz_shifted = fs_kin_vz - ctxt_dict["pv_coords"][
                        ..., 2
                    ].unsqueeze(1)
                else:
                    raise ValueError(
                        "No vertex z variable found in fs_vars for impact parameter recalculation"
                    )

                fs_dx = vx - ctxt_dict["pv_coords"][..., 0].unsqueeze(1)
                fs_dy = vy - ctxt_dict["pv_coords"][..., 1].unsqueeze(1)
                fs_dz = fs_kin_vz_shifted

                fs_sin_phi = fs_kin_feats[..., self.fs_var_idx_dict["sin_phi"]]
                fs_cos_phi = fs_kin_feats[..., self.fs_var_idx_dict["cos_phi"]]
                fs_eta = self.feature_scaler["eta"].inverse_transform(
                    fs_kin_feats[..., self.fs_var_idx_dict["eta"]]
                )

                d0 = -fs_dx * fs_sin_phi + fs_dy * fs_cos_phi
                z0 = fs_dz - torch.sinh(fs_eta) * (
                    fs_dx * fs_cos_phi + fs_dy * fs_sin_phi
                )
                ctxt_dict["pf_ctxt_data"] = torch.cat(
                    [
                        fs_kin_feats[..., :-1],
                        self.feature_scaler["vz_shifted"]
                        .transform(fs_kin_vz_shifted)
                        .unsqueeze(-1),
                        self.feature_scaler["d0"].transform(d0).unsqueeze(-1),
                        self.feature_scaler["z0"].transform(z0).unsqueeze(-1),
                        fs_class_raw,
                    ],
                    dim=-1,
                )

            fs_class = fs_class_raw.argmax(-1)
            # classes < 3 are charged particles
            ctxt_dict["fs_mask"] = (fs_class < 3) & (fs_mask)

            fs_impact_shape = (
                fs_part_shape[0],
                fs_part_shape[1],
                4,  # d0, d0Error, z0, z0Error
            )
            fs_impact = cast(
                torch.Tensor,
                self.impact_model.sample(
                    fs_impact_shape,
                    ctxt_dict,
                    to_cpu=False,
                ),
            )
            fs_impact = self.unscale_impact(fs_impact)
            # Undo masking for impact parameters
            ctxt_dict["fs_mask"] = fs_mask
        else:
            fs_impact = {}
        return (
            self.unscaler.unscale_variables(fs_part, ctxt_dict)
            | fs_impact
            | {"bad_idxs": bad_idxs}
        )
