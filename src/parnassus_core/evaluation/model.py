import yaml

import numpy as np
import torch
import torch.nn as nn

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.utils.scaling import FeatureScaler

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

    def extract_var(self, data, var_name, var_idx, sin_cos=False):
        shift = 0
        if var_name.endswith("phi") and sin_cos:
            ctxt_ = torch.atan2(
                data[..., var_idx],
                data[..., var_idx + 1],
            )
            shift = 2
        elif var_name.endswith("class"):
            ctxt_ = data[..., var_idx : var_idx + 5].argmax(-1)
            shift = 5
        else:
            ctxt_ = self.feature_scaler[var_name].inverse_transform(data[..., var_idx])
            shift = 1
        return ctxt_, shift

    def unscale_variables(self, fs_data, ctxt_dict):
        fs_data = fs_data.cpu()
        ctxt_dict = {k: v.cpu() for k, v in ctxt_dict.items()}
        new_data_dict = {}
        var_idx = 0
        for var in self.ctxt_global_vars:
            if var == "means":
                var_idx += len(
                    [
                        el
                        for el in self.ctxt_vars
                        if (el.startswith("truth_") and "class" not in el)
                    ]
                )
            elif var.startswith("truth_") or var.startswith("ntruth"):
                var = var.replace("truth_", "").replace("ntruth", "npart")
                new_data_dict[f"tr_{var}"] = (
                    self.feature_scaler[var]
                    .inverse_transform(ctxt_dict["ctxt_global_data"][..., var_idx])
                    .numpy()
                )
                var_idx += 1
            elif var.startswith("pflow_"):
                var = var.replace("pflow_", "")
                new_data_dict[f"pf_{var}"] = (
                    self.feature_scaler[var]
                    .inverse_transform(ctxt_dict["ctxt_global_data"][..., var_idx])
                    .numpy()
                )
                var_idx += 1
        var_idx = 0
        for var in self.ctxt_vars:
            if var.startswith("pflow_"):
                continue
            var_name = var.replace("truth_", "")
            ctxt_, shift = self.extract_var(
                ctxt_dict["ctxt_data"], var_name, var_idx, self.sin_cos
            )
            var_idx += shift
            ctxt_ = ctxt_.numpy()
            if var_name == "ptrel":
                ctxt_ = ctxt_ * new_data_dict["tr_ht"].reshape(-1, 1)
                var_name = "pt"
            new_data_dict[f"tr_{var_name}"] = ctxt_
        var_idx = 0
        for var in self.fs_vars:
            var_name = var.replace("pflow_", "").replace("npflow", "npart")
            fs_, shift = self.extract_var(fs_data, var_name, var_idx, self.sin_cos)
            var_idx += shift
            fs_ = fs_.numpy()
            if var_name == "ptrel":
                fs_ = fs_ * new_data_dict["pf_ht"].reshape(-1, 1)
                var_name = "pt"
            new_data_dict[f"fs_{var_name}"] = fs_
        if "ctxt_mask" in ctxt_dict:
            new_data_dict["ctxt_mask"] = ctxt_dict["ctxt_mask"].numpy()
        if "fs_mask" in ctxt_dict:
            new_data_dict["fs_mask"] = ctxt_dict["fs_mask"].numpy()
        if "eventNumber" in ctxt_dict:
            new_data_dict["eventNumber"] = ctxt_dict["eventNumber"].numpy()
        return new_data_dict

    def unscale_impact(self, fs_data):
        fs_data = fs_data.cpu()
        new_data_dict = {}
        var_idx = 0
        for var_name in ["d0", "d0Error", "z0", "z0Error"]:
            fs_, shift = self.extract_var(fs_data, var_name, var_idx, self.sin_cos)
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
            self.unscale_variables(fs_part, ctxt_dict)
            | fs_impact
            | {"bad_idxs": bad_idxs}
        )
