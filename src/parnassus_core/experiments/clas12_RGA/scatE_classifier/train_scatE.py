"""Train the scatE classifier: truth (p, theta, phi) of the scattered electron -> pflow_scatE.

Ported from the MLP section of MLP.ipynb. From the repository root:

    python -m parnassus_core.experiments.clas12_RGA.scatE_classifier.train_scatE \
        --root-file <parnassus_data>/v8/pre-filter_train_data_v8.root --n-events 10000000

The run writes the min-validation-loss checkpoint (``best_model.ckpt``), the diagnostic
plots (loss curves, confusion matrix, ROC) and the metrics into ``--outdir`` (default
``trained_models/classifier``). The evaluation chain reads the checkpoint named by the
``scatE_filter.ckpt_path`` key of its config (the paper checkpoint in
``pre-trained-models/``). Model and optimizer settings are in CONFIG; the data and run
options can be overridden on the command line (``--help``).
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import awkward as ak
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import uproot
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt  # noqa: E402


CONFIG = {
    # data
    "root_file": None,  # required: --root-file
    "tree": "evt_tree",
    "n_events": 29_800_000,  # events read from the front of the tree
    "val_frac": 0.15,
    "test_frac": 0.15,
    # model  (defaults = Optuna best trial: 4 x 128, lr 2.929e-3, dropout 0.02)
    "hidden": (128, 128, 128, 128),
    "dropout": 0.02,
    # optimization
    "lr": 2.929e-3,
    "n_epochs": 50,
    "patience": 10,  # stop early after this many epochs with no val-loss improvement (0/None = off)
    "batch_size": 8192,
    "eval_batch_size": 65536,
    # misc
    "seed": 0,
    "device": "auto",  # "auto" | "cuda" | "cpu"
    "threshold": 0.5,  # operating point for accuracy/F1/confusion matrix
    "outdir": "trained_models/classifier",
    "timestamp_outdir": False,  # append _YYYYmmdd-HHMMSS so runs never clobber
}

# structural, not hyperparameters
PARTICLE_FIELDS = ["p", "theta", "phi", "pdgId"]
INPUT_FIELDS = ["p", "theta", "phi"]


class Logger:
    """Print to stdout and tee everything into train_log.txt."""

    def __init__(self, path):
        self.fh = open(path, "w")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.fh.write(f"{msg}\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def load_events(cfg, log=print):
    """Read the tree and reduce it to (labels, scattered-electron features).

    Returns a structured numpy array of labels (fields scatE, eventNumber) and an
    awkward record array with fields (p, theta, phi), aligned event by event.
    """
    evt_tree = uproot.open(f"{cfg['root_file']}:{cfg['tree']}")
    n_events = min(cfg["n_events"], evt_tree.num_entries)
    log(f"reading {n_events} of {evt_tree.num_entries} events from {cfg['root_file']}")

    labels = evt_tree.arrays(
        ["scatE", "eventNumber"],
        aliases={"scatE": "pflow_scatE"},
        entry_stop=n_events,
    )
    truth_particles = evt_tree.arrays(
        PARTICLE_FIELDS,
        aliases={
            "p": "truth_p",
            "theta": "truth_theta",
            "phi": "truth_phi",
            "pdgId": "truth_pdgId",
        },
        entry_stop=n_events,
    )

    # only keep electrons
    electron_mask = truth_particles["pdgId"] == 11
    truth_electrons = ak.zip({f: truth_particles[f] for f in PARTICLE_FIELDS})[
        electron_mask
    ]

    # the scattered electron is the first one in the list, so events with no truth
    # electron have nothing to take - drop them from features AND labels together
    has_electron = ak.to_numpy(ak.num(truth_electrons, axis=1)) >= 1
    n_dropped = int(np.sum(~has_electron))
    if n_dropped:
        log(f"dropped {n_dropped} events with no truth electron")
    truth_electrons = truth_electrons[has_electron]
    pflow_labels = labels[has_electron].to_numpy()

    # keep only the model inputs of the scattered electron
    truth_electrons = truth_electrons[:, 0][INPUT_FIELDS]

    log(f"labels: shape {pflow_labels.shape} dtype {pflow_labels.dtype}")
    log(f"data: truth_electrons: {truth_electrons.type}")
    return pflow_labels, truth_electrons


def prepare_tensors(pflow_labels, truth_electrons, cfg, device, log=print):
    """Features/labels -> finite filter -> shuffle -> train/val/test -> standardized tensors."""
    rng = np.random.default_rng(cfg["seed"])

    # features: (N, 3) = truth level (p, theta, phi) of the scattered electron
    X = np.stack(
        [ak.to_numpy(truth_electrons[f]) for f in INPUT_FIELDS], axis=1
    ).astype(np.float32)
    # labels: (N,) = 0/1 whether the scattered electron was reconstructed
    y = pflow_labels["scatE"].astype(np.float32)

    # drop any event with a non-finite feature
    good = np.isfinite(X).all(axis=1)
    X, y = X[good], y[good]
    log(f"dropped {int(np.sum(~good))} events with non-finite features")
    log(f"X: {X.shape} {X.dtype}   y: {y.shape} {y.dtype}")
    class_balance = float(y.mean())
    log(
        f"class balance: scatE==1 -> {class_balance * 100:.2f}%,  scatE==0 -> {(1 - class_balance) * 100:.2f}%"
    )

    # shuffle, then train/val/test split
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_val = int(cfg["val_frac"] * len(X))
    n_test = int(cfg["test_frac"] * len(X))
    n_train = len(X) - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    # standardize the features using the training set only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    log(f"feature means: {mean}, feature stds: {std}")

    def to_tensors(Xa, ya):
        Xt = torch.from_numpy((Xa - mean) / std).to(device)
        yt = torch.from_numpy(ya).to(device)
        return Xt, yt

    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_val_t, y_val_t = to_tensors(X_val, y_val)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    log(
        f"train: {len(X_train_t)}   val: {len(X_val_t)}   test: {len(X_test_t)}   device: {device}"
    )
    splits = {
        "train": (X_train_t, y_train_t),
        "val": (X_val_t, y_val_t),
        "test": (X_test_t, y_test_t),
    }
    stats = {
        "mean": mean,
        "std": std,
        "class_balance": class_balance,
        "n_train": int(len(X_train_t)),
        "n_val": int(len(X_val_t)),
        "n_test": int(len(X_test_t)),
    }
    return splits, stats


class MLP(nn.Module):
    """Simple fully connected net: (p, theta, phi) -> single logit for scatE."""

    def __init__(self, n_inputs=3, hidden=(64, 64, 64), dropout=0.0):
        super().__init__()
        layers = []
        n_prev = n_inputs
        for n_hidden in hidden:
            layers.append(nn.Linear(n_prev, n_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n_prev = n_hidden
        layers.append(nn.Linear(n_prev, 1))  # raw logit, sigmoid is inside the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_logits(model, X_t, eval_batch_size=65536):
    """Run the model over a whole dataset in batches (no grad, eval mode)."""
    was_training = model.training
    model.eval()
    out = torch.empty(len(X_t), device=X_t.device)
    for start in range(0, len(X_t), eval_batch_size):
        stop = start + eval_batch_size
        out[start:stop] = model(X_t[start:stop])
    if was_training:
        model.train()
    return out


def train(model, splits, cfg, outdir, checkpoint_extra, log=print):
    """Train, tracking the minimum-validation-loss epoch and checkpointing it as we go."""
    X_train_t, y_train_t = splits["train"]
    X_val_t, y_val_t = splits["val"]
    device = X_train_t.device
    batch_size = cfg["batch_size"]

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    history = []
    best = {"epoch": None, "val_loss": float("inf"), "state": None}
    patience = cfg.get("patience") or 0  # 0 / None disables early stopping
    epochs_since_best = 0

    for epoch in range(1, cfg["n_epochs"] + 1):
        # ---- train ----
        model.train()
        order = torch.randperm(len(X_train_t), device=device)
        running_loss, n_seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(idx)
            n_seen += len(idx)
        train_loss = running_loss / n_seen

        # ---- validate ----
        val_logits = predict_logits(model, X_val_t, cfg["eval_batch_size"])
        val_loss = loss_fn(val_logits, y_val_t).item()
        val_acc = ((val_logits > 0).float() == y_val_t).float().mean().item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # ---- checkpoint the best epoch so far ----
        marker = ""
        if val_loss < best["val_loss"]:
            best = {
                "epoch": epoch,
                "val_loss": val_loss,
                "state": copy.deepcopy(model.state_dict()),
            }
            # written every time it improves, so a killed run still leaves a usable checkpoint
            torch.save(
                {
                    "model_state": best["state"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    **checkpoint_extra,
                },
                outdir / "best_model.ckpt",
            )
            marker = "  <- best"
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if patience:
                marker = f"  ({epochs_since_best}/{patience} since best)"

        log(
            f"epoch {epoch:3d}/{cfg['n_epochs']}  train loss {train_loss:.4f}  "
            f"val loss {val_loss:.4f}  val acc {val_acc:.4f}{marker}"
        )

        # ---- early stopping ----
        if patience and epochs_since_best >= patience:
            log(
                f"early stop: no val-loss improvement for {patience} epochs "
                f"(stopped at epoch {epoch} of {cfg['n_epochs']})"
            )
            break

    log(
        f"best epoch: {best['epoch']} (val loss {best['val_loss']:.4f}) -> {outdir / 'best_model.ckpt'}"
    )
    return history, best


def plot_loss_curves(history, best_epoch, outdir):
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], "-o", ms=3, label="train loss")
    ax.plot(
        epochs, [h["val_loss"] for h in history], "-o", ms=3, label="validation loss"
    )
    best = next(h for h in history if h["epoch"] == best_epoch)
    ax.plot(
        best_epoch,
        best["val_loss"],
        "*",
        ms=14,
        color="tab:red",
        label=f"best epoch {best_epoch} (val loss {best['val_loss']:.4f})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross entropy")
    ax.set_title("MLP training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["scatE=0", "scatE=1"]).plot(
        ax=axs[0], colorbar=False
    )
    axs[0].set_title("Counts")
    ConfusionMatrixDisplay(
        cm / cm.sum(axis=1, keepdims=True), display_labels=["scatE=0", "scatE=1"]
    ).plot(ax=axs[1], colorbar=False, values_format=".3f")
    axs[1].set_title("Normalized by truth row")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, test_probs, threshold, auc, outdir):
    fpr, tpr, thresholds = roc_curve(y_true, test_probs)

    # where the threshold used for the confusion matrix sits on the curve
    i_op = np.argmin(np.abs(thresholds - threshold))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, label=f"MLP (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess (AUC = 0.5)")
    ax.plot(
        fpr[i_op],
        tpr[i_op],
        "o",
        ms=7,
        color="tab:red",
        label=f"threshold = {threshold} (TPR {tpr[i_op]:.3f}, FPR {fpr[i_op]:.3f})",
    )
    ax.set_xlabel("False positive rate  (scatE=0 called 1)")
    ax.set_ylabel("True positive rate  (scatE=1 found)")
    ax.set_title("ROC curve - scatE classification (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close(fig)


def evaluate(model, splits, cfg, outdir, log=print):
    """Test-set accuracy / F1 / confusion matrix / ROC for the currently loaded weights."""
    X_test_t, y_test_t = splits["test"]
    threshold = cfg["threshold"]

    test_probs = (
        torch.sigmoid(predict_logits(model, X_test_t, cfg["eval_batch_size"]))
        .cpu()
        .numpy()
    )
    y_true = y_test_t.cpu().numpy().astype(int)
    y_pred = (test_probs > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, test_probs)

    # baseline for comparison: always predict the majority class
    majority = int(round(y_true.mean()))
    baseline_acc = accuracy_score(y_true, np.full_like(y_true, majority))

    log(f"test events: {len(y_true)}   threshold: {threshold}")
    log(f"accuracy: {acc:.4f}")
    log(f"f1 score: {f1:.4f}")
    log(f"AUC: {auc:.4f}")
    log(f"confusion matrix (rows = truth, cols = predicted):\n{cm}")
    log(f"majority-class baseline accuracy ({majority}): {baseline_acc:.4f}")

    plot_confusion_matrix(cm, outdir)
    plot_roc_curve(y_true, test_probs, threshold, auc, outdir)

    return {
        "test_events": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "majority_class": majority,
        "majority_baseline_accuracy": float(baseline_acc),
    }


def resolve_outdir(cfg):
    outdir = Path(cfg["outdir"])
    if cfg["timestamp_outdir"]:
        outdir = outdir.with_name(f"{outdir.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def parse_args(argv=None):
    """Command-line overrides for the data and run entries of CONFIG."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root-file", required=True, help="AllEvents ROOT file with evt_tree"
    )
    parser.add_argument("--tree", default=CONFIG["tree"])
    parser.add_argument(
        "--n-events",
        type=int,
        default=CONFIG["n_events"],
        help="events read from the front of the tree",
    )
    parser.add_argument("--n-epochs", type=int, default=CONFIG["n_epochs"])
    parser.add_argument(
        "--patience",
        type=int,
        default=CONFIG["patience"],
        help="early-stopping patience in epochs (0 = off)",
    )
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument(
        "--device", default=CONFIG["device"], choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument("--outdir", default=CONFIG["outdir"])
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="append _YYYYmmdd-HHMMSS to --outdir so runs never clobber",
    )
    args = parser.parse_args(argv)
    return {
        "root_file": args.root_file,
        "tree": args.tree,
        "n_events": args.n_events,
        "n_epochs": args.n_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "outdir": args.outdir,
        "timestamp_outdir": args.timestamp,
    }


def main(argv=None):
    cfg = {**CONFIG, **parse_args(argv)}
    outdir = resolve_outdir(cfg)
    log = Logger(outdir / "train_log.txt")
    try:
        log(f"output directory: {outdir.resolve()}")
        device = cfg["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg["seed"])

        t0 = time.time()
        pflow_labels, truth_electrons = load_events(cfg, log=log)
        splits, stats = prepare_tensors(
            pflow_labels, truth_electrons, cfg, device, log=log
        )
        log(f"data ready in {time.time() - t0:.1f}s")

        model = MLP(
            n_inputs=len(INPUT_FIELDS),
            hidden=tuple(cfg["hidden"]),
            dropout=cfg["dropout"],
        ).to(device)
        log(str(model))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"trainable parameters: {n_params}")

        with open(outdir / "config.json", "w") as fh:
            json.dump(
                {
                    **cfg,
                    "hidden": list(cfg["hidden"]),
                    "device": device,
                    "trainable_parameters": n_params,
                    "feature_mean": stats["mean"].tolist(),
                    "feature_std": stats["std"].tolist(),
                    **{
                        k: stats[k]
                        for k in ("class_balance", "n_train", "n_val", "n_test")
                    },
                },
                fh,
                indent=2,
            )

        # the normalization stats travel with the weights, so the checkpoint stands alone
        checkpoint_extra = {
            "config": {**cfg, "hidden": list(cfg["hidden"]), "device": device},
            "input_fields": INPUT_FIELDS,
            "feature_mean": stats["mean"],
            "feature_std": stats["std"],
        }

        t0 = time.time()
        history, best = train(model, splits, cfg, outdir, checkpoint_extra, log=log)
        dt = time.time() - t0
        log(
            f"trained {len(history)} of {cfg['n_epochs']} epochs in {dt:.1f}s "
            f"({dt / len(history):.2f}s/epoch)"
        )

        with open(outdir / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"]
            )
            writer.writeheader()
            writer.writerows(history)
        plot_loss_curves(history, best["epoch"], outdir)

        # evaluate the checkpoint we keep, not the last epoch
        model.load_state_dict(best["state"])
        metrics = evaluate(model, splits, cfg, outdir, log=log)
        metrics.update(
            best_epoch=best["epoch"],
            best_val_loss=best["val_loss"],
            epochs_run=len(history),
            stopped_early=len(history) < cfg["n_epochs"],
        )
        with open(outdir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        log(f"wrote {sorted(p.name for p in outdir.iterdir())}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
