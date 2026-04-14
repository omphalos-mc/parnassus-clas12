from typing import cast


import numpy as np
import torch
import torch.nn as nn

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.data.utils.scaling import FeatureScaler, Unscaler

from parnassus_core.evaluation.config import EvaluationConfig, ModelConfig
from parnassus_core.data.utils.variable_helper import VariableHelper

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

        self.feature_scaler = FeatureScaler(config.scale_dict)

        self.fs_vars = config.var_config.fs_vars
        self.ctxt_vars: VarList = config.var_config.ctxt_vars
        self.ctxt_global_vars: VarList = (
            config.var_config.ctxt_global_vars + self.fs_vars["event"]
        )

        self.variable_helper = VariableHelper(collection_names=["truth", "pflow"])

        self.ht_shift = self.feature_scaler["ht"].shift
        self.ht_scale = self.feature_scaler["ht"].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale
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

    def _get_fs_var_idxs(self) -> dict[str, int]:
        idx_dict = {}
        idx_dict["ht"] = self.fs_vars["event"].index(
            f"{self.unscaler.target_collection}_ht"
        )
        idx_dict["npart"] = self.fs_vars["event"].index(
            f"n{self.unscaler.target_collection}"
        )
        # Build a dict mapping variable names to their index in the particle feature vector
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
            elif var == "class":
                idx_dict[var] = curr_idx
                curr_idx += 5
            else:
                idx_dict[var] = curr_idx
                curr_idx += 1

        return idx_dict

    def compile_models(self):
        self.event_model = cast(FlowLightning, torch.compile(self.event_model))
        self.particle_model = cast(FlowLightning, torch.compile(self.particle_model))
        if self.impact_model is not None:
            self.impact_model = cast(FlowLightning, torch.compile(self.impact_model))

    def load_lightning_model(self, model_config: ModelConfig) -> FlowLightning:
        model = FlowLightning.load_from_checkpoint(
            model_config.ckpt_path,
            map_location=torch.device("cpu"),
        ).eval()
        model.change_sampler(
            model_config.sampler_type,
            model_config.n_steps,
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
                ctxt_dict,
                to_cpu=False,
            ),
        )
        fs_ht = fs_evt[..., self.fs_var_idx_dict["ht"]]
        fs_npart = self.feature_scaler["npart"].inverse_transform(
            fs_evt[..., self.fs_var_idx_dict["npart"]]
        )
        good_evt_mask = (
            (fs_npart > 0)
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
        ctxt_dict["ctxt_global_data"] = torch.cat(
            [ctxt_dict["ctxt_global_data"], fs_evt], -1
        )
        ctxt_dict["fs_mask"] = fs_mask
        fs_part_shape = (fs_evt.shape[0], self.max_particles, 12)
        fs_part = cast(
            torch.Tensor,
            self.particle_model.sample(
                fs_part_shape,
                ctxt_dict,
                to_cpu=False,
            ),
        )
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
