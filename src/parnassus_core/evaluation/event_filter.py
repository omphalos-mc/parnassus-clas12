"""Scattered-electron (scatE) pre-filter for the CLAS12 evaluation chain.

The CLAS12 event and particle flow models are trained only on events in which
the scattered electron is reconstructed, so evaluation must offer them the same
kind of events.  A small pre-trained MLP maps the truth scattered electron's
``(p, theta, phi)`` to ``P(pflow_scatE == 1)``, a learned reconstruction
efficiency, and events are kept by sampling that probability.

:class:`ScatESelector` is the selection, usable as an
:class:`~parnassus_core.data.loaders.EventSelector`: ``EagerDataset`` applies it
while reading, for ROOT and LUND input alike, so only the kept events are
loaded.  The ``scatE_filter:`` block of an evaluation config turns it on.

Modes
-----
``sample``     keep ~ Bernoulli(prob).  The default, and the only mode that is
               correct for fast simulation: the classifier is well calibrated, so
               sampling reproduces the true positive rate by construction.
``threshold``  keep = prob > threshold.  Deterministic, but at the 0.5 training
               operating point it keeps ~21.7% of events against a true rate of
               ~19.7% on the classifier's test split, i.e. it over-counts
               reconstructed electrons by ~10%.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
import torch.nn as nn
from rich import print as rprint

if TYPE_CHECKING:
    from parnassus_core.evaluation.config import ScatEFilterConfig

# The scattered electron is selected as the *first* truth electron, not the
# highest-momentum one, because that is what the classifier was trained on.
# (They agree in 99.99% of events, but "trained on" is the binding constraint.)
ELECTRON_PDGID = 11

# Feature order is fixed by the checkpoint's ``input_fields``; never reorder.
INPUT_FIELDS = ("p", "theta", "phi")

DEFAULT_CKPT = (
    "src/parnassus_core/experiments/clas12_RGA/pre-trained-models/pre-filter_model.ckpt"
)

# Truth branches the features are built from, in INPUT_FIELDS order after pdgId.
FEATURE_BRANCHES = ("truth_pdgId", "truth_p", "truth_theta", "truth_phi")
# Reconstruction label; present in full-simulation ROOT files, absent from LUND.
LABEL_BRANCH = "pflow_scatE"

SELECTION_MODES = ("sample", "threshold")


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


def first_electron_features(
    pdg_id: np.ndarray,
    p: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classifier features of each event's first truth electron.

    Parameters
    ----------
    pdg_id, p, theta, phi : np.ndarray
        Ragged per-event truth arrays (``dtype=object``, one array per event),
        particles in file order.

    Returns
    -------
    features : np.ndarray
        ``(n_events, 3)`` float32 ``(p, theta, phi)`` of the first electron;
        zero for events without a usable one.
    valid : np.ndarray
        Boolean mask of the events with an electron and finite features.
    """
    n_events = len(pdg_id)
    features = np.zeros((n_events, len(INPUT_FIELDS)), dtype=np.float32)
    valid = np.zeros(n_events, dtype=bool)
    if n_events == 0:
        return features, valid

    counts = np.fromiter(
        (len(event) for event in pdg_id), dtype=np.int64, count=n_events
    )
    flat_pdg_id = np.concatenate(pdg_id)
    electron_rows = np.flatnonzero(flat_pdg_id == ELECTRON_PDGID)
    if not len(electron_rows):
        return features, valid

    event_of_row = np.repeat(np.arange(n_events), counts)
    events, first = np.unique(event_of_row[electron_rows], return_index=True)
    rows = electron_rows[first]
    features[events] = np.stack(
        [np.concatenate(var)[rows] for var in (p, theta, phi)], axis=1
    ).astype(np.float32)
    valid[events] = np.isfinite(features[events]).all(axis=1)
    features[~valid] = 0.0
    return features, valid


def report_closure(
    n_read: int,
    n_no_electron: int,
    prob: np.ndarray,
    keep: np.ndarray,
    label: np.ndarray,
    mode: str,
) -> None:
    """Print the self-test: does the selection reproduce the true rate?

    Free to compute when the true ``pflow_scatE`` label ships in the file.
    """
    rprint("\n[bold blue]scatE filter closure report[/bold blue]")
    rprint(f"  events read              : {n_read}")
    rprint(f"  events without truth e-  : {n_no_electron}")
    rprint(f"  events kept              : {int(keep.sum())} ({keep.mean():.5f})")
    rprint(f"  mean predicted prob      : {prob.mean():.5f}")
    rprint(f"  true pflow_scatE rate    : {label.mean():.5f}")
    rprint(f"  mode                     : {mode}")

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
        " reference from scatE_filter.py (the pflow_scatE == 1 events), not from"
        " this selection.[/dim]"
    )


class ScatEClassifier:
    """The loaded scatE MLP with its feature standardization.

    Parameters
    ----------
    model : ScatEMLP
        The network, in eval mode.
    mean, std : np.ndarray
        Training-set feature mean and standard deviation.
    device : torch.device
        Device the network runs on.
    ckpt_path : str, Path or None
        Checkpoint the network came from, recorded for provenance.
    """

    def __init__(
        self,
        model: ScatEMLP,
        mean: np.ndarray,
        std: np.ndarray,
        device: torch.device,
        ckpt_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        self.device = device
        self.ckpt_path = ckpt_path

    @classmethod
    def load(cls, ckpt_path: str | Path, device: torch.device) -> Self:
        """Load the classifier from a checkpoint written by ``train_scatE.py``."""
        model, mean, std, _ = load_classifier(ckpt_path, device)
        return cls(model, mean, std, device, ckpt_path=ckpt_path)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``P(pflow_scatE == 1)`` for each ``(p, theta, phi)`` row."""
        return predict_proba(self.model, features, self.mean, self.std, self.device)


class ScatESelector:
    """Keep events by the scatE classifier's reconstruction probability.

    An :class:`~parnassus_core.data.loaders.EventSelector`: called on
    consecutive chunks of raw truth branches, it returns the keep mask and
    records the per-event probabilities for :meth:`report`.  Events without a
    truth electron (or with non-finite features) get probability 0 and are
    dropped.  ``sample`` mode draws one uniform number per event in file order,
    so for a given seed the kept events do not depend on the chunk size.

    Parameters
    ----------
    classifier : ScatEClassifier
        The loaded classifier.
    mode : {"sample", "threshold"}
        Selection mode (see the module docstring).
    threshold : float
        Probability cut of ``threshold`` mode.
    seed : int
        Seed of ``sample`` mode's random numbers.
    """

    branches: Sequence[str] = FEATURE_BRANCHES
    optional_branches: Sequence[str] = (LABEL_BRANCH,)

    def __init__(
        self,
        classifier: ScatEClassifier,
        *,
        mode: str = "sample",
        threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        if mode not in SELECTION_MODES:
            raise ValueError(
                f"Unknown scatE selection mode {mode!r}; expected one of "
                f"{SELECTION_MODES}."
            )
        self.classifier = classifier
        self.mode = mode
        self.threshold = threshold
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._prob: list[np.ndarray] = []
        self._keep: list[np.ndarray] = []
        self._valid: list[np.ndarray] = []
        self._label: list[np.ndarray] | None = []

    @classmethod
    def from_config(
        cls, config: ScatEFilterConfig, classifier: ScatEClassifier
    ) -> Self:
        """A selector with the mode, threshold and seed of a ``scatE_filter`` block."""
        return cls(
            classifier,
            mode=config.mode,
            threshold=config.threshold,
            seed=config.seed,
        )

    def __call__(self, chunk: dict[str, np.ndarray]) -> np.ndarray:
        pdg_id, p, theta, phi = (chunk[name] for name in FEATURE_BRANCHES)
        features, valid = first_electron_features(pdg_id, p, theta, phi)
        prob = np.zeros(len(valid), dtype=np.float32)
        if valid.any():
            prob[valid] = self.classifier.predict(features[valid])
        if self.mode == "sample":
            keep = self._rng.random(len(prob)) < prob
        else:
            keep = (prob > self.threshold) & valid

        self._prob.append(prob)
        self._keep.append(keep)
        self._valid.append(valid)
        if self._label is not None:
            if LABEL_BRANCH in chunk:
                self._label.append(np.asarray(chunk[LABEL_BRANCH]))
            else:
                self._label = None
        return keep

    @property
    def n_classified(self) -> int:
        """Events the classifier has seen so far."""
        return sum(len(prob) for prob in self._prob)

    @property
    def n_accepted(self) -> int:
        """Events accepted so far, before any cap on the number kept."""
        return sum(int(keep.sum()) for keep in self._keep)

    def describe(self) -> str:
        """One-line provenance of the selection, for output metadata."""
        cut = (
            f"seed={self.seed}"
            if self.mode == "sample"
            else f"threshold={self.threshold}"
        )
        return (
            f"{self.classifier.ckpt_path} mode={self.mode} {cut} "
            f"classified={self.n_classified} accepted={self.n_accepted}"
        )

    def report(self) -> None:
        """Print the selection summary.

        Covers every classified event: the selection pass reads whole chunks, so
        the last chunk can extend past the event at which the kept-event cap was
        reached.  When the input carries the ``pflow_scatE`` label (full
        simulation), the closure report against it is printed instead.
        """
        if not self._prob:
            rprint("[bold yellow]scatE filter: no events classified[/bold yellow]")
            return
        prob = np.concatenate(self._prob)
        keep = np.concatenate(self._keep)
        n_no_electron = int((~np.concatenate(self._valid)).sum())
        if self._label:
            report_closure(
                n_read=len(prob),
                n_no_electron=n_no_electron,
                prob=prob,
                keep=keep,
                label=np.concatenate(self._label),
                mode=self.mode,
            )
            return
        rprint("\n[bold blue]scatE filter summary[/bold blue]")
        rprint(f"  events classified        : {len(prob)}")
        rprint(f"  events without truth e-  : {n_no_electron}")
        rprint(f"  events accepted          : {int(keep.sum())} ({keep.mean():.5f})")
        rprint(f"  mean predicted prob      : {prob.mean():.5f}")
        rprint(f"  mode                     : {self.mode}")
