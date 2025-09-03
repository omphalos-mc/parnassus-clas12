import argparse

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import DataLoader
from tqdm import tqdm

from rich import print as rprint

from parnassus_core.experiments.cms_2011.dataset import CMSDataset

from parnassus_core.evaluation.config import EvaluationConfig
from parnassus_core.evaluation.model import EvaluationModel


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-bs", "--batch_size", type=int, default=64)
parser.add_argument("-ne", "--n_events", type=int)
parser.add_argument("--suffix", type=str, default="")

args = parser.parse_args()

torch.set_grad_enabled(False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EvaluationConfig.from_yaml(args.config)
    model = EvaluationModel(config)
    model.to(device)
    model.compile_models()

    for ds_config in config.data_config.datasets:
        rprint(f"[bold blue]Evaluating on dataset:[/bold blue] {ds_config.path}")
        eval_subpath = config.eval_path / ds_config.name
        dataset = CMSDataset(
            ds_config.path,
            config.data_config.max_particles,
            train_type="event",
            sin_cos=True,
            zero_neutral_vtx=True,
            scale_dict=config.scale_dict,
            var_dict=config.var_dict,
            reduce_ds=args.n_events if args.n_events else ds_config.n_events,
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
        fs_vars = model.fs_vars + ["ind"]
        if config.impact_model is not None:
            fs_vars += ["d0", "d0Error", "z0", "z0Error"]
        l_fs_data = {
            key.replace("pflow_", "").replace("ptrel", "pt"): np.zeros(
                (n_events, dataset.max_particles)
            )
            for key in fs_vars
        }
        l_eventNumber = np.zeros(n_events, dtype=np.int32)
        n = 0
        bad_idxs = []
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
            bad_idxs.extend(sampled_data.get("bad_idxs", []))

        for key in l_fs_data.keys():
            l_fs_data[key] = l_fs_data[key][:n]
            if key in l_tr_data:
                l_tr_data[key] = l_tr_data[key][:n]
        l_eventNumber = l_eventNumber[:n]
        eval_subpath = eval_subpath.with_stem(
            eval_subpath.stem + f"_eval_{(n // 1000):d}k{args.suffix}.root"
        )
        with uproot.recreate(eval_subpath) as file:
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
            file["metadata/event_model"] = (
                f"{config.event_model.ckpt_path} {config.event_model.n_steps}"
            )
            file["metadata/particle_model"] = (
                f"{config.particle_model.ckpt_path} {config.particle_model.n_steps}"
            )
            if config.impact_model:
                file["metadata/impact_model"] = (
                    f"{config.impact_model.ckpt_path} {config.impact_model.n_steps}"
                )
        if len(bad_idxs) > 0:
            rprint(f"[bold red]Found {len(bad_idxs)} bad events:[/bold red] {bad_idxs}")
        rprint(f"[bold green]Done:[/bold green] Saved to {eval_subpath.absolute()}")


if __name__ == "__main__":
    main()
