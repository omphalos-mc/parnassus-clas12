import argparse
import os
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import DataLoader
from tqdm import tqdm

from rich import print as rprint

from parnassus_core.data import EagerDataset

from parnassus_core.evaluation.config import EvaluationConfig
from parnassus_core.evaluation.model import EvaluationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-ne", "--n_events", type=int)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes assembling batches while the GPU samples.",
    )
    return parser.parse_args()


def limit_cpu_threads(num_workers: int, max_threads: int = 8) -> int:
    """Size torch's CPU thread pool to leave room for the DataLoader workers.

    By default torch starts one intra-op thread per CPU in the affinity mask.
    When busy DataLoader workers share those CPUs, every small parallel op
    then waits on a descheduled thread: unscaling one 4096-event batch went
    from ~90 ms to ~12 s on 26 CPUs with two busy processes. Eight threads was
    the fastest setting measured. An explicit ``OMP_NUM_THREADS`` is respected.

    Parameters
    ----------
    num_workers : int
        DataLoader worker processes that will share this process's CPUs.
    max_threads : int, optional
        Upper bound on the pool size.

    Returns
    -------
    int
        The resulting ``torch.get_num_threads()``.
    """
    if "OMP_NUM_THREADS" not in os.environ:
        free_cpus = len(os.sched_getaffinity(0)) - num_workers
        torch.set_num_threads(max(1, min(max_threads, free_cpus)))
    return torch.get_num_threads()


def write_eval_output(
    path: Path,
    pflow: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    event_number: np.ndarray,
    config: EvaluationConfig,
) -> None:
    """Write generated events as jagged ``pflow`` / ``truth`` records.

    Each ``(n_events, max_particles)`` column is stored as a variable-length
    float32 list per event (always ``max_particles`` long), with ``ind``
    marking the real particles.
    """

    def to_record(columns: dict[str, np.ndarray]) -> ak.Array:
        return ak.zip(
            {
                key: ak.from_regular(ak.from_numpy(values.astype(np.float32)))
                for key, values in columns.items()
            }
        )

    with uproot.recreate(path) as file:
        file["evt_tree"] = {
            "pflow": to_record(pflow),
            "truth": to_record(truth),
            "eventNumber": ak.Array(event_number),
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


def main():
    args = parse_args()
    limit_cpu_threads(args.num_workers)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EvaluationConfig.from_yaml(args.config)
    model = EvaluationModel(config)
    model.to(device)
    model.compile_models()

    for ds_config in config.data_config.datasets:
        rprint(f"[bold blue]Evaluating on dataset:[/bold blue] {ds_config.path}")
        eval_subpath = config.eval_path / ds_config.name
        dataset = EagerDataset(
            ds_config.path,
            max_particles=config.data_config.max_particles,
            train_type="event",
            sin_cos=True,
            zero_neutral_vtx=False,
            scale_dict=config.scale_dict,
            var_dict=config.var_dict,
            reduce_ds=args.n_events if args.n_events else ds_config.n_events,
            entry_start=0,
            mode="eval",
            recalculate_ip=config.recalculate_ip,
            eta_clamp=config.data_config.eta_clamp,
            class_based_scaling=config.data_config.class_based_scaling,
        )
        loader = DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        n_events = len(dataset)
        l_tr_data = {
            key.replace("truth_", "")
            .replace("ptrel", "pt")
            .replace("_shifted", ""): np.zeros((n_events, dataset.max_particles))
            for key in model.ctxt_vars + ["ind"]
        }
        fs_vars = model.fs_vars["particle"] + ["ind"]
        if config.impact_model is not None:
            fs_vars += model.fs_vars["impact"]
        l_fs_data = {
            key.replace("pflow_", "")
            .replace("ptrel", "pt")
            .replace("_shifted", ""): np.zeros((n_events, dataset.max_particles))
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
            batch_size = len(sampled_data["eventNumber"])
            for key in l_tr_data:
                if key != "ind":
                    l_tr_data[key][n : n + batch_size] = sampled_data["truth_" + key]
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
        for key in l_tr_data.keys():
            l_tr_data[key] = l_tr_data[key][:n]
        l_eventNumber = l_eventNumber[:n]
        eval_subpath = eval_subpath.with_stem(
            eval_subpath.stem + f"_eval_{(n // 1000):d}k{args.suffix}.root"
        )
        write_eval_output(eval_subpath, l_fs_data, l_tr_data, l_eventNumber, config)
        if len(bad_idxs) > 0:
            rprint(f"[bold red]Found {len(bad_idxs)} bad events:[/bold red] {bad_idxs}")
        rprint(f"[bold green]Done:[/bold green] Saved to {eval_subpath.absolute()}")


if __name__ == "__main__":
    main()
