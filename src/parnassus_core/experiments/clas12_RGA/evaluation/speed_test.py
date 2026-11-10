"""Time every stage of the CLAS12 fast-simulation inference chain on one GPU.

Runs the chain of ``scripts/run_evaluation_chain.sh`` (minus postprocessing)
and reports wall time and throughput per stage::

    scatE pre-filter   AllEvents ROOT -> read -> features -> classifier
                       -> Bernoulli select -> write filtered ROOT
    dataset load       filtered ROOT -> EagerDataset (ROOT read, preprocessing)
    generation loop    batch assembly -> host->device -> event net -> glue
                       -> particle net -> unscale + device->host -> collect
    output             ROOT write

Each stage follows the production code path. The pre-filter loop is
``scatE_filter.main`` with a timer around each step. The generation loop is
``evaluation/eval.py``, with ``EvaluationModel``'s ``event_model.sample``,
``particle_model.sample`` and ``unscaler.unscale_variables`` wrapped in timers;
"glue" is whatever ``generate_event`` spends outside those three (the
empty-event veto, count rounding, masking). CUDA is synchronized at every stage
boundary so each stage is charged for its own kernels.

Two deliberate departures from the production chain:

* The dataset stage loads the config's filtered file -- what eval.py reads --
  rather than the file this script's pre-filter just wrote, so its read time is
  a cold CFS read as in production, not a page-cache hit. The pre-filter and
  eval outputs go to a scratch ``--work-dir`` and are deleted afterwards.
* ``--warmup-batches`` batches are generated and discarded before the timed
  loop, so the loop measures steady state. The warm-up is reported as one-time
  setup, alongside checkpoint loading.

The report is printed and saved, with a JSON copy of the raw numbers, to
``<eval_path>/logs/speed_test_<timestamp>.{log,json}``.

Example
-------
::

    python -m parnassus_core.experiments.clas12_RGA.evaluation.speed_test \\
        -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/config.yaml \\
        --filter-input ../parnassus_data/v8/v8_test.root
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias
from unittest import mock

import awkward as ak
import numpy as np
import torch
import uproot
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from parnassus_core.data import EagerDataset
from parnassus_core.data.loaders.root_loader import ROOTFileLoader
from parnassus_core.evaluation.config import EvaluationConfig
from parnassus_core.evaluation.eval import limit_cpu_threads, write_eval_output
from parnassus_core.evaluation.model import EvaluationModel
from parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter import (
    DEFAULT_CKPT,
    INPUT_FIELDS,
    ScatEMLP,
    build_features,
    load_classifier,
    predict_proba,
)

TREE_NAME = "evt_tree"

_EXHAUSTED = object()


@dataclass
class Stage:
    """Accumulated wall time and event count of one timed stage."""

    seconds: float = 0.0
    events: int = 0
    calls: int = 0


class StageTimer:
    """Accumulate wall time per named stage, synchronizing CUDA at each boundary.

    Synchronizing on entry and exit charges each stage for its own kernels,
    instead of whichever later stage first blocks on them.

    Parameters
    ----------
    device : torch.device
        Device the chain runs on; synchronization is skipped off CUDA.
    """

    def __init__(self, device: torch.device) -> None:
        self.stages: dict[str, Stage] = {}
        self._is_cuda = device.type == "cuda"

    def _sync(self) -> None:
        if self._is_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def __call__(self, name: str, events: int = 0) -> Iterator[Stage]:
        """Time the enclosed block under ``name``, crediting it ``events``.

        Yields the stage so a caller that only learns the event count inside
        the block can add it there.
        """
        stage = self.stages.setdefault(name, Stage())
        self._sync()
        start = time.perf_counter()
        yield stage
        self._sync()
        stage.seconds += time.perf_counter() - start
        stage.events += events
        stage.calls += 1

    def seconds(self, name: str) -> float:
        """Total time of ``name``, or 0 if it never ran."""
        stage = self.stages.get(name)
        return stage.seconds if stage is not None else 0.0

    def events(self, name: str) -> int:
        """Total events of ``name``, or 0 if it never ran."""
        stage = self.stages.get(name)
        return stage.events if stage is not None else 0

    def instrument(
        self,
        stack: ExitStack,
        owner: Any,
        attr: str,
        name: str,
        count: Callable[..., int] | None = None,
    ) -> None:
        """Time every call of ``owner.attr`` under ``name`` until ``stack`` closes.

        ``count`` receives the call's arguments and returns the events it
        processes.
        """
        original = getattr(owner, attr)

        def timed(*args, **kwargs):
            events = count(*args, **kwargs) if count is not None else 0
            with self(name, events):
                return original(*args, **kwargs)

        stack.enter_context(mock.patch.object(owner, attr, timed))

    def instrument_iter(
        self, stack: ExitStack, owner: Any, attr: str, name: str
    ) -> None:
        """Time each ``next()`` on the iterators ``owner.attr`` returns.

        For lazy readers such as ``tree.iterate``, where the work happens when
        the next chunk is pulled rather than when the iterator is created.
        """
        original = getattr(owner, attr)

        def timed(*args, **kwargs):
            items = original(*args, **kwargs)
            while True:
                with self(name):
                    item = next(items, _EXHAUSTED)
                if item is _EXHAUSTED:
                    return
                yield item

        stack.enter_context(mock.patch.object(owner, attr, timed))


@dataclass
class GeneratedEvents:
    """Generated events in eval.py's in-memory layout, ready to write."""

    truth: dict[str, np.ndarray]
    pflow: dict[str, np.ndarray]
    event_number: np.ndarray


class Row(NamedTuple):
    """One report line. ``in_chain`` rows get a share of the chain time."""

    label: str
    seconds: float
    events: int = 0
    in_chain: bool = True


Section: TypeAlias = tuple[str, list[Row]]


def time_prefilter(
    input_path: Path,
    output_path: Path,
    classifier: tuple[ScatEMLP, np.ndarray, np.ndarray],
    *,
    n_read: int,
    n_keep: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
    timer: StageTimer,
) -> tuple[int, int]:
    """Run ``scatE_filter --mode sample`` with each step timed.

    Same loop as ``scatE_filter.main``: read every branch chunk by chunk, keep
    events by Bernoulli-sampling the classifier probability, append the kept
    events to ``output_path``, and stop after the chunk that reaches
    ``n_keep``. The closure report is skipped, as it is diagnostics rather than
    part of the chain.

    Returns
    -------
    tuple[int, int]
        Events read and events kept.
    """
    model, mean, std = classifier
    rng = np.random.default_rng(seed)
    n_total = 0
    n_kept = 0
    tree_created = False

    with timer("prefilter/total") as total:
        with timer("prefilter/read"):
            tree = uproot.open(f"{input_path}:{TREE_NAME}")
            branches = list(tree.keys())
            entry_stop = min(tree.num_entries, n_read)
            chunks = tree.iterate(branches, step_size=chunk_size, entry_stop=entry_stop)

        progress = tqdm(total=entry_stop, unit="evt", desc="Pre-filter")
        with uproot.recreate(output_path) as fout:
            while n_kept < n_keep:
                with timer("prefilter/read") as read:
                    chunk = next(chunks, None)
                if chunk is None:
                    break
                n_chunk = len(chunk)
                read.events += n_chunk

                with timer("prefilter/features", n_chunk):
                    features, valid = build_features(chunk)

                with timer("prefilter/classifier", n_chunk):
                    prob = np.zeros(n_chunk, dtype=np.float32)
                    if valid.any():
                        prob[valid] = predict_proba(
                            model, features[valid], mean, std, device
                        )

                with timer("prefilter/select", n_chunk):
                    keep = rng.random(n_chunk) < prob
                    kept = {name: chunk[name][keep] for name in branches}
                    kept["scatE_prob"] = prob[keep]

                with timer("prefilter/write", n_chunk):
                    if not tree_created:
                        fout.mktree(
                            TREE_NAME,
                            {
                                **{name: ak.type(chunk[name]) for name in branches},
                                "scatE_prob": np.float32,
                            },
                        )
                        tree_created = True
                    if keep.any():
                        fout[TREE_NAME].extend(kept)  # type: ignore[union-attr]

                n_total += n_chunk
                n_kept += int(keep.sum())
                progress.update(n_chunk)
        progress.close()
    total.events = n_total
    return n_total, n_kept


def time_dataset_load(
    config: EvaluationConfig, n_events: int, timer: StageTimer
) -> EagerDataset:
    """Build the eval dataset exactly as eval.py does, timing the ROOT reads.

    Everything ``EagerDataset.__init__`` spends outside ``ROOTFileLoader``'s
    reads is preprocessing (flattening, phi wrapping, derived event variables,
    per-event means).
    """
    ds_config = config.data_config.datasets[0]
    with ExitStack() as stack:
        timer.instrument(stack, ROOTFileLoader, "load_branches", "dataset/read")
        timer.instrument_iter(stack, ROOTFileLoader, "iter_branches", "dataset/read")
        with timer("dataset/total") as total:
            dataset = EagerDataset(
                ds_config.path,
                max_particles=config.data_config.max_particles,
                train_type="event",
                sin_cos=True,
                zero_neutral_vtx=False,
                scale_dict=config.scale_dict,
                var_dict=config.var_dict,
                reduce_ds=n_events,
                entry_start=0,
                mode="eval",
                recalculate_ip=config.recalculate_ip,
                eta_clamp=config.data_config.eta_clamp,
                class_based_scaling=config.data_config.class_based_scaling,
            )
    total.events = timer.stages["dataset/read"].events = len(dataset)
    return dataset


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def time_generation(
    model: EvaluationModel,
    dataset: EagerDataset,
    *,
    batch_size: int,
    num_workers: int,
    warmup_batches: int,
    device: torch.device,
    timer: StageTimer,
) -> GeneratedEvents:
    """Run eval.py's generation loop over ``dataset`` with each stage timed.

    The first ``warmup_batches`` batches are generated untimed-by-stage (as
    ``setup/warmup``) and discarded; the timed loop then covers every batch,
    those included.
    """
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )

    with timer("setup/warmup"):
        warmup_batches_iter = iter(loader)
        for _ in range(min(warmup_batches, len(loader))):
            model.generate_event(_to_device(next(warmup_batches_iter), device))
        del warmup_batches_iter

    # Host-side buffers, keyed and filled exactly as in eval.py.
    n_events = len(dataset)
    truth = {
        key.replace("truth_", "")
        .replace("ptrel", "pt")
        .replace("_shifted", ""): np.zeros((n_events, dataset.max_particles))
        for key in model.ctxt_vars + ["ind"]
    }
    pflow = {
        key.replace("pflow_", "")
        .replace("ptrel", "pt")
        .replace("_shifted", ""): np.zeros((n_events, dataset.max_particles))
        for key in model.fs_vars["particle"] + ["ind"]
    }
    event_number = np.zeros(n_events, dtype=np.int32)

    n = 0
    with ExitStack() as stack:
        timer.instrument(
            stack,
            model.event_model,
            "sample",
            "generate/event_net",
            count=lambda shape, *_, **__: shape[0],
        )
        timer.instrument(
            stack,
            model.particle_model,
            "sample",
            "generate/particle_net",
            count=lambda shape, *_, **__: shape[0],
        )
        timer.instrument(
            stack,
            model.unscaler,
            "unscale_variables",
            "generate/unscale",
            count=lambda fs_data, *_, **__: fs_data.shape[0],
        )
        with timer("generate/total", n_events):
            batches = iter(loader)
            for _ in tqdm(range(len(loader)), desc="Generating"):
                with timer("generate/batch_assembly") as assembly:
                    batch = next(batches)
                n_in = len(batch["ctxt_global_data"])
                assembly.events += n_in

                with timer("generate/h2d", n_in):
                    batch = _to_device(batch, device)

                with timer("generate/generate_event", n_in):
                    sampled = model.generate_event(batch)
                n_out = len(sampled["eventNumber"])

                with timer("generate/collect", n_out):
                    for key in truth:
                        if key != "ind":
                            truth[key][n : n + n_out] = sampled["truth_" + key]
                        else:
                            truth[key][n : n + n_out] = sampled["ctxt_mask"] > 0
                    for key in pflow:
                        if key != "ind":
                            pflow[key][n : n + n_out] = sampled["fs_" + key]
                        else:
                            pflow[key][n : n + n_out] = sampled["fs_mask"] > 0
                    event_number[n : n + n_out] = sampled["eventNumber"]
                n += n_out

    return GeneratedEvents(
        truth={key: values[:n] for key, values in truth.items()},
        pflow={key: values[:n] for key, values in pflow.items()},
        event_number=event_number[:n],
    )


def _git_revision() -> str:
    """Short HEAD hash, marked ``-dirty`` when tracked files are modified."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if status else head


def _format_rate(seconds: float, events: int) -> tuple[str, str]:
    """``(events/s, µs/event)`` cells, or dashes for a stage without events."""
    if events <= 0 or seconds <= 0:
        return "-", "-"
    return f"{events / seconds:,.0f}", f"{1e6 * seconds / events:,.1f}"


def build_report(
    timer: StageTimer,
    config: EvaluationConfig,
    meta: dict[str, Any],
    counts: dict[str, int],
    wall_seconds: float,
) -> tuple[list[Section], dict[str, float]]:
    """Assemble the report rows, one section per chain stage, and a summary.

    Section totals are measured around the whole section, so "other" is the
    time the section spent outside its named stages.
    """
    sec = timer.seconds
    n_read, n_loaded, n_generated = (
        counts["prefilter_read"],
        counts["dataset_loaded"],
        counts["generated"],
    )

    n_warmup = meta["warmup_batches"]
    setup = [
        Row("CUDA context init", sec("setup/cuda_init"), in_chain=False),
        Row(
            "Flow model load (event + particle)",
            sec("setup/flow_models"),
            in_chain=False,
        ),
        Row("scatE classifier load", sec("setup/classifier"), in_chain=False),
        Row(
            f"Warm-up ({n_warmup} batch{'' if n_warmup == 1 else 'es'})",
            sec("setup/warmup"),
            in_chain=False,
        ),
    ]

    prefilter_total = sec("prefilter/total")
    prefilter = [
        Row("ROOT read (all branches)", sec("prefilter/read"), n_read),
        Row("Feature build", sec("prefilter/features"), n_read),
        Row("Classifier MLP", sec("prefilter/classifier"), n_read),
        Row("Bernoulli selection", sec("prefilter/select"), n_read),
        Row("ROOT write (kept events)", sec("prefilter/write"), n_read),
    ]
    prefilter_other = prefilter_total - sum(row.seconds for row in prefilter)
    prefilter += [
        Row("other", max(0.0, prefilter_other)),
        Row("total", prefilter_total, n_read),
    ]

    dataset_read = sec("dataset/read")
    dataset_total = sec("dataset/total")
    dataset = [
        Row("ROOT read", dataset_read, n_loaded),
        Row("Preprocessing", dataset_total - dataset_read, n_loaded),
        Row("total", dataset_total, n_loaded),
    ]

    event_net = sec("generate/event_net")
    particle_net = sec("generate/particle_net")
    unscale = sec("generate/unscale")
    generate_event = sec("generate/generate_event")
    glue = generate_event - event_net - particle_net - unscale
    loop_total = sec("generate/total")
    loop_named = (
        sec("generate/batch_assembly")
        + sec("generate/h2d")
        + generate_event
        + sec("generate/collect")
    )
    event_cfg, particle_cfg = config.event_model, config.particle_model
    generation = [
        Row(
            "Batch assembly (__getitem__ + collate)",
            sec("generate/batch_assembly"),
            timer.events("generate/batch_assembly"),
        ),
        Row("Host -> device", sec("generate/h2d"), timer.events("generate/h2d")),
        Row(
            f"Event network ({event_cfg.sampler_type}, {event_cfg.n_steps} steps)",
            event_net,
            timer.events("generate/event_net"),
        ),
        Row("Event -> particle glue (veto, rounding)", glue, n_loaded),
        Row(
            f"Particle network ({particle_cfg.sampler_type}, "
            f"{particle_cfg.n_steps} steps)",
            particle_net,
            timer.events("generate/particle_net"),
        ),
        Row("Unscale + device -> host", unscale, timer.events("generate/unscale")),
        Row(
            "Collect into host arrays",
            sec("generate/collect"),
            timer.events("generate/collect"),
        ),
        Row("other", max(0.0, loop_total - loop_named)),
        Row("total", loop_total, n_loaded),
    ]

    output = [Row("ROOT write", sec("output/write"), timer.events("output/write"))]

    networks = event_net + particle_net
    chain = prefilter_total + dataset_total + loop_total + sec("output/write")
    summary_rows = [
        Row("Networks only (event + particle)", networks, n_generated),
        Row("Full chain, excl. one-time setup", chain, n_generated),
        Row(
            "End-to-end wall time, incl. setup",
            wall_seconds,
            n_generated,
            in_chain=False,
        ),
    ]

    sections = [
        ("One-time setup", setup),
        (f"scatE pre-filter  ({n_read:,} events read)", prefilter),
        (f"Dataset load  ({n_loaded:,} events)", dataset),
        (f"Generation loop  (batch size {meta['batch_size']:,})", generation),
        (f"Output  ({n_generated:,} events)", output),
        (f"Summary  (per generated event, {n_generated:,} events)", summary_rows),
    ]
    summary = {
        "networks_seconds": networks,
        "chain_seconds": chain,
        "wall_seconds": wall_seconds,
        "networks_events_per_second": n_generated / networks if networks else 0,
        "chain_events_per_second": n_generated / chain if chain else 0,
        "wall_events_per_second": n_generated / wall_seconds if wall_seconds else 0,
    }
    return sections, summary


def print_report(
    console: Console,
    sections: list[Section],
    meta: dict[str, Any],
    counts: dict[str, int],
    chain_seconds: float,
) -> None:
    """Print the run configuration, then one table row per timed stage."""
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    # Wrap rather than truncate: the checkpoint paths are the run's provenance.
    info.add_column(overflow="fold")
    for key, value in meta.items():
        info.add_row(key, escape(str(value)))
    console.rule("[bold blue]Parnassus inference speed test")
    console.print(info)
    console.print(
        f"\nevents: {counts['prefilter_read']:,} read by the pre-filter, "
        f"{counts['prefilter_kept']:,} kept; {counts['dataset_loaded']:,} loaded "
        f"for generation, {counts['generated']:,} generated "
        f"({counts['dataset_loaded'] - counts['generated']:,} vetoed by "
        "generate_event)"
    )

    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("Stage")
    for header in ("Time (s)", "Events", "Events/s", "µs/event", "Share"):
        table.add_column(header, justify="right")
    last_index = len(sections) - 1
    for index, (title, rows) in enumerate(sections):
        if index:
            table.add_section()
        table.add_row(f"[bold]{escape(title)}[/bold]")
        for row in rows:
            rate, per_event = _format_rate(row.seconds, row.events)
            share = (
                f"{100 * row.seconds / chain_seconds:.1f}%"
                if row.in_chain and chain_seconds
                else ""
            )
            is_bold = row.label == "total" or index == last_index
            table.add_row(
                f"  {escape(row.label)}",
                f"{row.seconds:,.2f}",
                f"{row.events:,}" if row.events else "",
                rate,
                per_event,
                share,
                style="bold" if is_bold else "",
            )
    console.print(table)
    console.print(
        "[dim]Share is the fraction of the full chain excluding one-time setup. "
        "Each stage's Events/s is over the events that stage processed.[/dim]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time each stage of the CLAS12 fast-simulation inference chain."
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--filter-input",
        type=str,
        required=True,
        help="Unfiltered AllEvents ROOT file the scatE pre-filter reads.",
    )
    parser.add_argument(
        "-ne",
        "--n-events",
        type=int,
        default=None,
        help="Events to generate. Default: the config's first dataset n_events.",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--filter-ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument(
        "--n-read",
        type=int,
        default=2_700_000,
        help="Cap on events the pre-filter reads.",
    )
    parser.add_argument(
        "--n-keep",
        type=int,
        default=None,
        help="Pre-filter stops after the chunk reaching this many kept events. "
        "Default: --n-events.",
    )
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Where the transient filter/eval ROOT outputs go. "
        "Default: $SCRATCH/parnassus_speed_test.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep the transient ROOT outputs instead of deleting them.",
    )
    parser.add_argument(
        "--skip-write", action="store_true", help="Skip the final ROOT write."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Where the report is saved. Default: <eval_path>/logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wall_start = time.perf_counter()
    limit_cpu_threads(args.num_workers)
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = StageTimer(device)

    config = EvaluationConfig.from_yaml(args.config)
    n_events = args.n_events or config.data_config.datasets[0].n_events
    n_keep = args.n_keep or n_events
    log_dir = Path(args.log_dir) if args.log_dir else config.eval_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    scratch_root = Path(
        args.work_dir
        or Path(os.environ.get("SCRATCH", tempfile.gettempdir()))
        / "parnassus_speed_test"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="speed_test_", dir=scratch_root))

    try:
        with timer("setup/cuda_init"):
            torch.zeros(1, device=device)
        with timer("setup/flow_models"):
            model = EvaluationModel(config)
            model.to(device)
            model.compile_models()
        with timer("setup/classifier"):
            classifier, mean, std, _ = load_classifier(args.filter_ckpt, device)
            # First-call cuBLAS setup belongs to setup, not to the first chunk.
            predict_proba(
                classifier,
                np.zeros((1024, len(INPUT_FIELDS)), dtype=np.float32),
                mean,
                std,
                device,
            )

        n_read, n_kept = time_prefilter(
            Path(args.filter_input),
            work_dir / "filtered.root",
            (classifier, mean, std),
            n_read=args.n_read,
            n_keep=n_keep,
            chunk_size=args.chunk_size,
            seed=args.seed,
            device=device,
            timer=timer,
        )
        dataset = time_dataset_load(config, n_events, timer)
        events = time_generation(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_batches=args.warmup_batches,
            device=device,
            timer=timer,
        )
        if not args.skip_write:
            with timer("output/write", len(events.event_number)):
                write_eval_output(
                    work_dir / "eval.root",
                    events.pflow,
                    events.truth,
                    events.event_number,
                    config,
                )
        wall_seconds = time.perf_counter() - wall_start
    finally:
        if not args.keep_outputs:
            shutil.rmtree(work_dir, ignore_errors=True)

    counts = {
        "prefilter_read": n_read,
        "prefilter_kept": n_kept,
        "dataset_loaded": len(dataset),
        "generated": len(events.event_number),
    }
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "cpus": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "git": _git_revision(),
        "config": args.config,
        "event_ckpt": str(config.event_model.ckpt_path),
        "particle_ckpt": str(config.particle_model.ckpt_path),
        "eval_dataset": str(config.data_config.datasets[0].path),
        "filter_input": args.filter_input,
        "filter_ckpt": args.filter_ckpt,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "trim_padding": config.trim_padding,
        "torch_compile": config.torch_compile,
        "warmup_batches": args.warmup_batches,
        "write": not args.skip_write,
        "work_dir": str(work_dir) + ("" if args.keep_outputs else " (deleted)"),
    }
    if device.type == "cuda":
        meta["peak_gpu_mem_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 2
        )

    sections, summary = build_report(timer, config, meta, counts, wall_seconds)
    console = Console(record=True, width=110)
    print_report(console, sections, meta, counts, summary["chain_seconds"])

    stem = f"speed_test_{datetime.now():%Y%m%d-T%H%M%S}"
    log_path = log_dir / f"{stem}.log"
    json_path = log_dir / f"{stem}.json"
    console.save_text(str(log_path))
    with open(json_path, "w") as file:
        json.dump(
            {
                "meta": meta,
                "counts": counts,
                "summary": summary,
                "stages": {name: asdict(stage) for name, stage in timer.stages.items()},
            },
            file,
            indent=2,
        )
    for path in (log_path, json_path):
        console.print(f"[bold green]Saved:[/bold green] {path}", soft_wrap=True)


if __name__ == "__main__":
    main()
