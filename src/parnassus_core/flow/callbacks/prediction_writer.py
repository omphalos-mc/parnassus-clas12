from pathlib import Path
import torch
from lightning import Callback, Trainer
import uproot
import awkward as ak

from parnassus_core.flow.utils.scaling import Unscaler


class PredictionWriter(Callback):
    def __init__(self):
        super().__init__()
        self.test_outputs = []
        self.is_setted_up = False

    def setup(self, trainer, pl_module, stage):
        if stage != "test":
            return

        self.trainer = trainer
        self.pl_module = pl_module

        self.eval_output_name = self.trainer.datamodule.eval_output_name
        self.eval_path = Path(self.trainer.log_dir)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pf_data, ctxt_dict = batch
        return_dict = {}
        with torch.no_grad():
            fs_data = pl_module.sample(
                pf_data.shape,
                ctxt_dict=ctxt_dict,
            )
        for key, val in ctxt_dict.items():
            return_dict[key] = val.cpu()
        return_dict["fs_data"] = fs_data.cpu()
        return_dict["pf_data"] = pf_data.cpu()
        self.test_outputs.append(return_dict)

    def on_test_epoch_end(self, trainer, pl_module):
        ctxt_dict = {}
        fs_data = []
        pf_data = []
        for el in self.test_outputs:
            for key, val in el.items():
                if key == "fs_data":
                    fs_data.append(val)
                elif key == "pf_data":
                    pf_data.append(val)
                else:
                    if key not in ctxt_dict:
                        ctxt_dict[key] = []
                    ctxt_dict[key].append(val)
        fs_data = torch.cat(fs_data, dim=0)
        pf_data = torch.cat(pf_data, dim=0)
        for key, val in ctxt_dict.items():
            ctxt_dict[key] = torch.cat(val, dim=0)
        out_dict = self.unscaler.unscale_variables(fs_data, ctxt_dict, pf_data=pf_data)
        out_dict["tr_mask"] = ctxt_dict["ctxt_mask"] > 0
        out_dict["pf_mask"] = ctxt_dict["fs_mask"] > 0
        n_events = len(out_dict["eventNumber"])
        eval_subpath = (
            self.eval_path / f"{self.eval_output_name}_{n_events // 1e3}k.root"
        )
        if eval_subpath.exists():
            print("Warning: Overwriting existing file:", eval_subpath)
        mapping = {
            "fs_": "fastsim",
            "pf_": "pflow",
            "tr_": "truth",
        }
        uproot_dict = {
            map_value: {
                key.replace(map_key, "").replace("mask", "ind"): ak.Array(
                    out_dict[key].tolist()
                )
                for key in out_dict
                if key.startswith(map_key)
            }
            for map_key, map_value in mapping.items()
        }
        with uproot.recreate(eval_subpath) as file:
            file["evt_tree"] = uproot_dict | {
                "eventNumber": ak.Array(out_dict["eventNumber"]),
            }
            file["metadata/model"] = (
                f"{self.trainer.ckpt_path} {self.pl_module.n_steps}"
            )
        print(
            f"Saved predictions to {eval_subpath.absolute()}. "
            f"Number of events: {n_events}"
        )
        self.test_outputs.clear()

    def on_test_epoch_start(self, trainer: Trainer, pl_module):
        if not self.is_setted_up:
            test_dataset = trainer.test_dataloaders.dataset
            self.unscaler = Unscaler(
                feature_scaler=test_dataset.feature_scaler,
                fs_vars=test_dataset.fs_vars,
                ctxt_vars=test_dataset.ctxt_vars,
                ctxt_global_vars=test_dataset.ctxt_global_vars,
                sin_cos=test_dataset.sin_cos,
            )
            self.is_setted_up = True
        self.test_outputs.clear()
