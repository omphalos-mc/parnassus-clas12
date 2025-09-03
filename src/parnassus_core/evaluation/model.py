import yaml

import numpy as np
import torch
import torch.nn as nn

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.utils.scaling import FeatureScaler, Unscaler

from parnassus_core.evaluation.config import EvaluationConfig, ModelConfig


def reshape_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


VarList = list[str]


class EvaluationModel(nn.Module):
    def __init__(self, config: EvaluationConfig):
        super().__init__()

        self.event_model = self.load_lightning_model(config.event_model)
        self.particle_model = self.load_lightning_model(config.particle_model)
        if config.impact_model is not None:
            self.impact_model = self.load_lightning_model(config.impact_model)
        else:
            self.impact_model = None

        self.feature_scaler = FeatureScaler(config.scale_dict)

        with open(config.var_dict, "r") as f:
            var_dict = yaml.safe_load(f)
        self.fs_vars: VarList = var_dict["fs_vars"]
        self.ctxt_vars: VarList = var_dict["ctxt_vars"]
        self.ctxt_global_vars: VarList = var_dict["ctxt_global_vars"]

        self.ht_shift = self.feature_scaler["ht"].shift
        self.ht_scale = self.feature_scaler["ht"].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale
        self.max_particles = config.data_config.max_particles
        self.sin_cos = True

        self.unscaler = Unscaler(
            feature_scaler=self.feature_scaler,
            ctxt_vars=self.ctxt_vars,
            fs_vars=self.fs_vars,
            ctxt_global_vars=self.ctxt_global_vars,
            sin_cos=self.sin_cos,
        )

    def compile_models(self):
        self.event_model = torch.compile(self.event_model)
        self.particle_model = torch.compile(self.particle_model)
        if self.impact_model is not None:
            self.impact_model = torch.compile(self.impact_model)

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

    def unscale_impact(self, fs_data):
        fs_data = fs_data.cpu()
        new_data_dict = {}
        var_idx = 0
        for var_name in ["d0", "d0Error", "z0", "z0Error"]:
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
            4,
        )
        fs_evt = self.event_model.sample(
            event_shape,
            ctxt_dict,
            to_cpu=False,
        )
        fs_npart = self.feature_scaler["npart"].inverse_transform(fs_evt[..., 3])
        fs_ht = fs_evt[..., 0]
        good_evt_mask = (
            (fs_npart > 0) & (fs_ht > self.min_ht_scaled) & (fs_ht < self.max_particles)
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
        fs_part = self.particle_model.sample(
            fs_part_shape,
            ctxt_dict,
            to_cpu=False,
        )
        if self.impact_model is not None:
            ctxt_dict["pf_ctxt_data"] = fs_part
            fs_class = fs_part[..., -5:].argmax(-1)
            ctxt_dict["fs_mask"] = (fs_class < 3) & (fs_mask)
            fs_impact_shape = (
                fs_part_shape[0],
                fs_part_shape[1],
                4,  # d0, d0Error, z0, z0Error
            )
            fs_impact = self.impact_model.sample(
                fs_impact_shape,
                ctxt_dict,
                to_cpu=False,
            )
            fs_impact = self.unscale_impact(fs_impact)
            ctxt_dict["fs_mask"] = fs_mask
        else:
            fs_impact = {}
        return (
            self.unscaler.unscale_variables(fs_part, ctxt_dict)
            | fs_impact
            | {"bad_idxs": bad_idxs}
        )
