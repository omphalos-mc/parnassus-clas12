import argparse
import yaml

import awkward as ak
import numpy as np
import torch
import uproot
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.utils.scaling import FeatureScaler
from parnassus_core.experiments.cms_2011.dataset import CMSDataset


def reshape_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


VarList = list[str]


class EvalModel(nn.Module):
    def __init__(self, config, max_particles=400):
        super().__init__()

        self.config = config
        self.event_model = self.load_lightning_model(config["event_model"])
        self.particle_model = self.load_lightning_model(config["particle_model"])
        if "impact_model" in config:
            self.impact_model = self.load_lightning_model(config["impact_model"])
        else:
            self.impact_model = None

        self.feature_scaler = FeatureScaler(config["scale_dict"])

        with open(config["var_dict"], "r") as f:
            var_dict = yaml.safe_load(f)
        self.fs_vars: VarList = var_dict["fs_vars"]
        self.ctxt_vars: VarList = var_dict["ctxt_vars"]
        self.ctxt_global_vars: VarList = var_dict["ctxt_global_vars"]

        self.ht_shift = self.feature_scaler["ht"].shift
        self.ht_scale = self.feature_scaler["ht"].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale
        self.max_particles = max_particles
        self.sin_cos = True

    def load_lightning_model(self, model_config) -> FlowLightning:
        model = FlowLightning.load_from_checkpoint(
            model_config["ckpt_path"],
            map_location=torch.device("cpu"),
        ).eval()
        model.change_sampler(
            model_config.get("sampler_type", "euler"), model_config.get("n_steps", 25)
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
        # idxs = torch.arange(fs_evt.shape[0], device=fs_evt.device).to(torch.int32)
        # good_idxs = idxs[good_evt_mask]
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
        return self.unscale_variables(fs_part, ctxt_dict)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-bs", "--batch_size", type=int, default=64)
parser.add_argument("-ne", "--n_events", type=int, default=1000)
parser.add_argument("--test_path", type=str, default=None, required=True)
parser.add_argument("--prefix", type=str, default="")

args = parser.parse_args()


def main():
    with open(args.config, "r") as fp:
        config = yaml.full_load(fp)

    model = EvalModel(config["model_configs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model)
    model.eval()
    model: EvalModel

    torch.set_grad_enabled(False)

    dataset = CMSDataset(
        args.test_path,
        400,
        train_type="event",
        sin_cos=True,
        zero_neutral_vtx=True,
        scale_dict=config["model_configs"]["scale_dict"],
        var_dict=config["model_configs"]["var_dict"],
        reduce_ds=args.n_events,
        entry_start=0,
        mode="eval",
    )
    loader = DataLoader(
        dataset,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )
    n_events = len(dataset)
    l_tr_data = {
        key.replace("truth_", "").replace("ptrel", "pt"): np.zeros(
            (n_events, dataset.max_particles)
        )
        for key in model.ctxt_vars + ["ind"]
    }
    l_fs_data = {
        key.replace("pflow_", "").replace("ptrel", "pt"): np.zeros(
            (n_events, dataset.max_particles)
        )
        for key in model.fs_vars + ["ind"]
    }
    l_eventNumber = np.zeros(n_events, dtype=np.int32)
    n = 0

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        sampled_data = model.generate_event(batch)
        batch_size = None
        for key in l_tr_data:
            if key != "ind":
                if batch_size is None:
                    batch_size = sampled_data["tr_" + key].shape[0]
                l_tr_data[key][n : n + batch_size] = sampled_data["tr_" + key]
            else:
                l_tr_data[key][n : n + batch_size] = sampled_data["ctxt_mask"] > 0
        for key in l_fs_data:
            if key != "ind":
                l_fs_data[key][n : n + batch_size] = sampled_data["fs_" + key]
            else:
                l_fs_data[key][n : n + batch_size] = sampled_data["fs_mask"] > 0
        l_eventNumber[n : n + batch_size] = sampled_data["eventNumber"]
        n += batch_size

    for key in l_fs_data.keys():
        l_fs_data[key] = l_fs_data[key][:n]
        if key in l_tr_data:
            l_tr_data[key] = l_tr_data[key][:n]
    l_eventNumber = l_eventNumber[:n]
    eval_path = "eval_data.root"
    with uproot.recreate(eval_path) as file:
        file["evt_tree"] = {
            "pflow": ak.zip(
                {
                    key: ak.Array(l_fs_data[key].astype(np.float32).tolist())
                    for key in l_fs_data.keys()
                }
            ),
            "truth": ak.zip(
                {
                    key: ak.Array(l_tr_data[key].astype(np.float32).tolist())
                    for key in l_tr_data.keys()
                }
            ),
            "eventNumber": ak.Array(l_eventNumber),
        }

    print(f"Saved to {eval_path}")


if __name__ == "__main__":
    main()
