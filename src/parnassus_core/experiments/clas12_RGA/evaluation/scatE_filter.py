"""Pre-filter a CLAS12 ``AllEvents`` ROOT file down to the events in which the
scattered electron is reconstructed.

The clas12_RGA event/particle flow models are trained on the ``AllScatE`` sample
(``pflow_scatE == 1`` everywhere, no ``npflow == 0`` events), so the full-chain
evaluation can only be run on data where the scattered electron is already known
to have been reconstructed.  This script supplies the missing selection: a small
pre-trained MLP maps the truth scattered-electron ``(p, theta, phi)`` to
``P(pflow_scatE == 1)`` -- a learned reconstruction efficiency -- and events are
kept by sampling that probability.  The result is a drop-in dataset for the
existing chain::

    AllEvents.root -> scatE_filter.py -> <stem>_sample.root -> eval.py -> postprocess_eval.py
                                      -> <stem>_truth.root  ------------------^  (-d reference)

Nothing in ``parnassus_core.evaluation`` / ``.data`` needs to change: the output
is the same flat ``evt_tree`` layout as the input, with every source branch
copied verbatim plus the classifier's per-event ``scatE_prob``.  The true
``pflow_scatE`` label is carried through too, so a kept event's status is always
recoverable downstream.

Modes
-----
``sample``     keep ~ Bernoulli(prob).  The default, and the only mode that is
               correct for fast simulation: the classifier is well calibrated, so
               sampling reproduces the true positive rate by construction.
``threshold``  keep = prob > --threshold.  Deterministic, but at the 0.5 training
               operating point it keeps ~21.7% of events against a true rate of
               ~19.7% on the classifier's test split, i.e. it over-counts
               reconstructed electrons by ~10%.
``truth``      keep = pflow_scatE == 1, ignoring the classifier.  Produces the
               honest reference sample for ``postprocess_eval -d`` (see the
               module docstring of that script for the eventNumber join).

Example
-------
::

    python -m parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter \
        -i ../parnassus_data/v8/v8_test.root \
        -o ../parnassus_data/v8/v8_test_scatE_sample.root \
        --mode sample --n-events 3_000_000 --seed 0
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import torch.nn as nn
import uproot
from rich import print as rprint
from tqdm import tqdm

# The scattered electron is selected as the *first* truth electron, not the
# highest-momentum one, because that is what the classifier was trained on.
# (They agree in 99.99% of events, but "trained on" is the binding constraint.)
ELECTRON_PDGID = 11

# Feature order is fixed by the checkpoint's ``input_fields``; never reorder.
INPUT_FIELDS = ("p", "theta", "phi")

DEFAULT_CKPT = "trained_models/classifier/best_model.ckpt"

# Branches needed to build the classifier features and the truth-mode mask.
FEATURE_BRANCHES = ["truth_pt", "truth_theta", "truth_phi", "truth_pdgId"]
LABEL_BRANCH = "pflow_scatE"


class ScatEMLP(nn.Module):
    """``(p, theta, phi)`` -> a single logit for ``pflow_scatE``.

    Deliberately *not* ``flow.modules.dense.DenseNetwork``: the checkpoint's
    state dict keys are positional inside a plain ``nn.Sequential``
    (``net.{0,3,6,9,12}``), with the gaps occupied by Dropout at 2/5/8/11.  The
    module must be built with ``dropout > 0`` or the indices shift to 0,2,4,6,8
    and ``load_state_dict`` fails.
    """

    def __init__(
        self,
        n_inputs: int = 3,
        hidden: tuple[int, ...] = (128,) * 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        n_prev = n_inputs
        for n_hidden in hidden:
            layers += [nn.Linear(n_prev, n_hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n_prev = n_hidden
        layers.append(nn.Linear(n_prev, 1))  # raw logit; the sigmoid lived in the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_classifier(
    ckpt_path: str | Path, device: torch.device
) -> tuple[ScatEMLP, np.ndarray, np.ndarray, dict]:
    """Load the checkpoint and its training-set standardization statistics."""
    # weights_only=False: the checkpoint carries numpy arrays and dicts.
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    if tuple(ckpt["input_fields"]) != INPUT_FIELDS:
        raise ValueError(
            f"checkpoint input_fields {tuple(ckpt['input_fields'])} != {INPUT_FIELDS}"
        )
    model = ScatEMLP(
        n_inputs=len(ckpt["input_fields"]),
        hidden=tuple(ckpt["config"]["hidden"]),
        dropout=ckpt["config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # required -- dropout must be off
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    return model, mean, std, ckpt


def build_features(chunk: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """Truth scattered-electron ``(p, theta, phi)`` for one chunk of events.

    Returns the ``(n_events, 3)`` feature array and a boolean mask marking the
    events for which a usable feature vector exists.  Rows without a truth
    electron, or with non-finite features (``p = pt / sin(theta)`` blows up at
    tiny theta), are zero-filled and flagged False -- the caller assigns them
    probability 0.
    """
    n_events = len(chunk)
    is_electron = chunk["truth_pdgId"] == ELECTRON_PDGID
    electrons = ak.zip(
        {
            "pt": chunk["truth_pt"][is_electron],
            "theta": chunk["truth_theta"][is_electron],
            "phi": chunk["truth_phi"][is_electron],
        }
    )
    has_electron = ak.to_numpy(ak.num(electrons, axis=1)) >= 1

    features = np.zeros((n_events, len(INPUT_FIELDS)), dtype=np.float32)
    if not has_electron.any():
        return features, has_electron

    # The scattered electron is index 0 of the truth electrons.
    scat_e = electrons[has_electron][:, 0]
    pt = ak.to_numpy(scat_e["pt"])
    theta = ak.to_numpy(scat_e["theta"])
    phi = ak.to_numpy(scat_e["phi"])  # raw radians, as trained
    with np.errstate(divide="ignore", invalid="ignore"):
        p = pt / np.sin(theta)
    features[has_electron] = np.stack([p, theta, phi], axis=1).astype(np.float32)

    valid = has_electron.copy()
    valid[has_electron] = np.isfinite(features[has_electron]).all(axis=1)
    features[~valid] = 0.0
    return features, valid


@torch.no_grad()
def predict_proba(
    model: ScatEMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    """P(pflow_scatE == 1) for each row of ``features``."""
    scaled = torch.from_numpy(((features - mean) / std).astype(np.float32)).to(device)
    logits = torch.empty(len(scaled), device=device)
    for start in range(0, len(scaled), batch_size):
        logits[start : start + batch_size] = model(scaled[start : start + batch_size])
    return torch.sigmoid(logits).float().cpu().numpy()


def report_closure(
    n_read: int,
    n_no_electron: int,
    prob: np.ndarray,
    keep: np.ndarray,
    label: np.ndarray,
    mode: str,
) -> None:
    """Print the self-test: does the selection reproduce the true rate?

    Free to compute, because the true ``pflow_scatE`` label ships in the file.
    """
    rprint("\n[bold blue]scatE filter closure report[/bold blue]")
    rprint(f"  events read              : {n_read}")
    rprint(f"  events without truth e-  : {n_no_electron}")
    rprint(f"  events kept              : {int(keep.sum())} ({keep.mean():.5f})")
    rprint(f"  mean predicted prob      : {prob.mean():.5f}")
    rprint(f"  true pflow_scatE rate    : {label.mean():.5f}")
    rprint(f"  mode                     : {mode}")

    if mode == "truth":
        return

    # Classifier metrics against the true label, at the *realized* operating
    # point.  Only threshold mode is comparable to the checkpoint's test-split
    # numbers in scatE_classifier/README.md (F1 ~0.896, accuracy ~0.957):
    # Bernoulli sampling is deliberately not the argmax, so it trades F1 for a
    # correct positive rate, which is what fast simulation needs.  ROC-AUC is
    # threshold-free and so comparable in every mode (~0.977).
    truth_pos = label > 0
    tp = int((keep & truth_pos).sum())
    fp = int((keep & ~truth_pos).sum())
    fn = int((~keep & truth_pos).sum())
    tn = int((~keep & ~truth_pos).sum())
    accuracy = (tp + tn) / max(len(keep), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    # rprint escapes: square brackets are rich markup, so spell the layout out.
    rprint(f"  confusion tn / fp / fn / tp    : {tn} / {fp} / {fn} / {tp}")
    rprint(
        f"  accuracy / precision / recall  : {accuracy:.5f} / {precision:.5f} / {recall:.5f}"
    )
    rprint(f"  F1 (at the {mode} point)   : {f1:.5f}")
    try:
        from sklearn.metrics import roc_auc_score

        rprint(f"  ROC-AUC (threshold-free)       : {roc_auc_score(label, prob):.5f}")
    except ImportError:  # pragma: no cover - sklearn is in the env, but stay optional
        rprint("  ROC-AUC                        : skipped (sklearn unavailable)")

    rprint(
        "  [dim]Purity of the kept sample is the precision above; the rest are"
        " false positives.\n  This is why postprocess_eval should take its -d"
        " reference from a --mode truth run, not from this file.[/dim]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a CLAS12 AllEvents ROOT file on the scatE classifier."
    )
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tree", type=str, default="evt_tree")
    parser.add_argument(
        "--mode", type=str, default="sample", choices=["sample", "threshold", "truth"]
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-ne",
        "--n-events",
        type=int,
        default=None,
        help="Input events to read (not output events). Default (or -1): the whole file.",
    )
    parser.add_argument(
        "--n-keep",
        type=int,
        default=None,
        help="Soft target: stop after the chunk in which this many events have "
        "been kept. Chunks are never split, so the closure report stays exact.",
    )
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("--output must differ from --input")

    tree = uproot.open(f"{input_path}:{args.tree}")
    branches = list(tree.keys())
    for required in [*FEATURE_BRANCHES, LABEL_BRANCH, "eventNumber"]:
        if required not in branches:
            raise KeyError(f"branch '{required}' missing from {input_path}")

    entry_start = args.entry_start
    entry_stop = tree.num_entries
    # A negative count means the whole file; passed through, -1 would become an
    # uproot entry_stop of -1, which silently drops the last event.
    if args.n_events is not None and args.n_events >= 0:
        entry_stop = min(entry_stop, entry_start + args.n_events)

    # Truth mode ignores the classifier for the selection; it only fills the
    # informational scatE_prob column, so a missing checkpoint is not fatal there.
    model = None
    if args.mode == "truth" and not Path(args.ckpt).exists():
        rprint(
            f"[bold yellow]Classifier {args.ckpt} not found; truth mode writes"
            " scatE_prob = NaN[/bold yellow]"
        )
    else:
        model, mean, std, ckpt = load_classifier(args.ckpt, device)
        rprint(
            f"[bold blue]Classifier:[/bold blue] {args.ckpt} "
            f"(epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.5f})"
        )
    rprint(
        f"[bold blue]Mode:[/bold blue] {args.mode}  [bold blue]device:[/bold blue] {device}"
    )
    rprint(
        f"[bold blue]Reading:[/bold blue] {input_path} entries "
        f"[{entry_start}, {entry_stop}) of {tree.num_entries}"
    )

    rng = np.random.default_rng(args.seed)
    all_prob: list[np.ndarray] = []
    all_keep: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    n_read = 0
    n_kept = 0
    n_no_electron = 0
    n_chunks_written = 0
    tree_created = False

    n_total = entry_stop - entry_start
    with uproot.recreate(output_path) as fout:
        progress = tqdm(total=n_total, unit="evt", desc="Filtering")
        for chunk in tree.iterate(
            branches,
            step_size=args.chunk_size,
            entry_start=entry_start,
            entry_stop=entry_stop,
        ):
            n_chunk = len(chunk)
            features, valid = build_features(chunk)
            prob = np.zeros(n_chunk, dtype=np.float32)
            if model is None:
                prob[:] = np.nan
            elif valid.any():
                prob[valid] = predict_proba(model, features[valid], mean, std, device)
            label = ak.to_numpy(chunk[LABEL_BRANCH])

            if args.mode == "sample":
                # No usable feature vector -> no prediction -> prob 0 -> dropped.
                keep = rng.random(n_chunk) < prob
            elif args.mode == "threshold":
                keep = (prob > args.threshold) & valid
            else:
                keep = label > 0

            n_read += n_chunk
            n_kept += int(keep.sum())
            n_no_electron += int((~valid).sum())
            all_prob.append(prob)
            all_keep.append(keep)
            all_label.append(label)

            if not tree_created:
                # Declare the branches explicitly with mktree.  Plain
                # ``fout[name] = dict`` writes an RNTuple in uproot >= 5.7, but
                # the input is a TTree and downstream ROOT-based tooling expects
                # one, so the output must stay a TTree.  Types come from the
                # unfiltered chunk, so an all-rejected first chunk still gives a
                # correctly typed (empty) tree.
                #
                # mktree gives every jagged branch its own counter
                # (``ntruth_pt``, ``npflow_theta``, ...) rather than sharing the
                # source's single ``ntruth`` / ``npflow``; uproot rejects a
                # shared counter.  The duplicates are dead weight but harmless:
                # ``ntruth`` / ``npflow`` are still copied verbatim, and every
                # reader here requests branches by exact name.
                fout.mktree(
                    args.tree,
                    {
                        **{name: ak.type(chunk[name]) for name in branches},
                        "scatE_prob": np.float32,
                    },
                )
                tree_created = True

            if keep.any():
                out = {name: chunk[name][keep] for name in branches}
                out["scatE_prob"] = prob[keep]
                fout[args.tree].extend(out)  # type: ignore[union-attr]
                n_chunks_written += 1

            progress.update(n_chunk)
            if args.n_keep is not None and n_kept >= args.n_keep:
                break
        progress.close()

    if n_chunks_written == 0:
        raise RuntimeError(
            f"no events passed the filter over entries "
            f"[{entry_start}, {entry_stop}) -- {output_path} holds an empty tree"
        )

    report_closure(
        n_read=n_read,
        n_no_electron=n_no_electron,
        prob=np.concatenate(all_prob),
        keep=np.concatenate(all_keep),
        label=np.concatenate(all_label),
        mode=args.mode,
    )
    rprint(
        f"\n[bold green]Done:[/bold green] {n_kept} events -> {output_path.absolute()}"
    )


if __name__ == "__main__":
    main()
