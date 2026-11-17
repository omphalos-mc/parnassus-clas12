from pathlib import Path

import numpy as np
import pytest

from parnassus_core.data import EagerDataset
from parnassus_core.evaluation.config import ScatEFilterConfig
from parnassus_core.evaluation.event_filter import (
    DEFAULT_CKPT,
    LABEL_BRANCH,
    ScatESelector,
    first_electron_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAS12 = REPO_ROOT / "src/parnassus_core/experiments/clas12_RGA"


def _ragged(parts, dtype=np.float64):
    out = np.empty(len(parts), dtype=object)
    for index, part in enumerate(parts):
        out[index] = np.asarray(part, dtype=dtype)
    return out


class FixedClassifier:
    """Stand-in for ScatEClassifier: probability = p / 10, clipped to [0, 1]."""

    ckpt_path = "stub.ckpt"

    def __init__(self):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return np.clip(features[:, 0] / 10.0, 0.0, 1.0).astype(np.float32)


def _chunk(n_events, seed=0):
    """Ragged truth chunk; every event but the multiples of 5 holds electrons."""
    rng = np.random.default_rng(seed)
    pdg_id, p, theta, phi = [], [], [], []
    for event in range(n_events):
        n = 1 + event % 3
        ids = rng.choice([22, 211, -211], size=n)
        if event % 5:
            ids[rng.integers(n)] = 11
        pdg_id.append(ids)
        p.append(rng.uniform(0.1, 9.0, size=n))
        theta.append(rng.uniform(0.1, 0.6, size=n))
        phi.append(rng.uniform(-np.pi, np.pi, size=n))
    return {
        "truth_pdgId": _ragged(pdg_id, np.int32),
        "truth_p": _ragged(p),
        "truth_theta": _ragged(theta),
        "truth_phi": _ragged(phi),
    }


def _split(chunk, at):
    return {k: v[:at] for k, v in chunk.items()}, {k: v[at:] for k, v in chunk.items()}


def test_first_electron_features_takes_the_first_electron():
    pdg_id = _ragged([[22, 11, 11], [211], [11], []], np.int32)
    p = _ragged([[1.0, 2.0, 3.0], [4.0], [np.inf], []])
    theta = _ragged([[0.1, 0.2, 0.3], [0.4], [0.5], []])
    phi = _ragged([[0.0, 0.5, 1.0], [1.5], [2.0], []])
    features, valid = first_electron_features(pdg_id, p, theta, phi)
    assert valid.tolist() == [True, False, False, False]
    assert features[0] == pytest.approx([2.0, 0.2, 0.5])
    assert not features[1:].any()


def test_sampling_is_seeded_and_chunk_size_invariant():
    chunk = _chunk(200)
    whole = ScatESelector(FixedClassifier(), seed=3)(chunk)
    first, second = _split(chunk, 77)
    selector = ScatESelector(FixedClassifier(), seed=3)
    split = np.concatenate([selector(first), selector(second)])
    assert np.array_equal(whole, split)
    assert selector.n_classified == 200
    assert selector.n_accepted == int(whole.sum())
    assert not np.array_equal(whole, ScatESelector(FixedClassifier(), seed=4)(chunk))


def test_events_without_an_electron_are_dropped():
    chunk = _chunk(100)
    for mode in ("sample", "threshold"):
        keep = ScatESelector(FixedClassifier(), mode=mode, threshold=0.0)(chunk)
        assert not keep[::5].any()


def test_threshold_mode():
    chunk = _chunk(100)
    keep = ScatESelector(FixedClassifier(), mode="threshold", threshold=0.5)(chunk)
    features, valid = first_electron_features(
        *(chunk[k] for k in ScatESelector.branches)
    )
    assert np.array_equal(keep, valid & (features[:, 0] / 10.0 > 0.5))


def test_report_uses_the_label_when_present(capsys):
    chunk = _chunk(50)
    chunk[LABEL_BRANCH] = (np.arange(50) % 2).astype(np.int32)
    selector = ScatESelector(FixedClassifier())
    selector(chunk)
    selector.report()
    assert "closure report" in capsys.readouterr().out
    assert "stub.ckpt mode=sample seed=0 classified=50" in selector.describe()


def test_config_rejects_truth_mode():
    with pytest.raises(ValueError, match="scatE_filter.py"):
        ScatEFilterConfig(ckpt_path="unused.ckpt", mode="truth")
    with pytest.raises(ValueError, match="Invalid scatE_filter mode"):
        ScatEFilterConfig(ckpt_path="unused.ckpt", mode="other")


def test_config_accepts_the_shipped_checkpoint(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    config = ScatEFilterConfig(ckpt_path=DEFAULT_CKPT, mode="threshold", threshold=0.4)
    assert config.ckpt_path.exists()


def _lund_line(index, pid, px, py, pz, mass):
    return f"{index} 0. 1 {pid} 0 0 {px} {py} {pz} 0.0 {mass} 0.0 0.0 0.0"


def test_eager_dataset_on_lund_input_with_selector(tmp_path):
    header = "{n} 1 1 0.0 1.0 11 10.600 2212 1 1.0"
    events = [
        # 0: e-, pi+, gamma, n -> Ng = 1, Nn = 1
        [
            _lund_line(1, 11, -0.4, -1.2, 7.2, 0.0005),
            _lund_line(2, 211, 0.3, 0.1, 0.7, 0.1396),
            _lund_line(3, 22, 0.1, 0.1, 0.8, 0.0),
            _lund_line(4, 2112, -0.2, 0.3, 1.0, 0.9396),
        ],
        # 1: no electron -> dropped by the selector
        [_lund_line(1, 22, 0.1, 0.2, 0.9, 0.0)],
        # 2: e-, two gammas -> Ng = 2
        [
            _lund_line(1, 11, 0.5, 0.5, 4.0, 0.0005),
            _lund_line(2, 22, 0.2, -0.1, 0.6, 0.0),
            _lund_line(3, 22, -0.1, 0.3, 0.9, 0.0),
        ],
    ]
    lines = []
    for particles in events:
        lines += [header.format(n=len(particles)), *particles]
    path = tmp_path / "sample.dat"
    path.write_text("\n".join(lines) + "\n")

    class HasElectron:
        branches = ("truth_pdgId",)
        optional_branches = ()

        def __call__(self, chunk):
            return np.array([11 in event for event in chunk["truth_pdgId"]])

    dataset = EagerDataset(
        str(path),
        var_dict=str(CLAS12 / "evaluation/configs/eval_vars.yaml"),
        scale_dict=str(CLAS12 / "configs/var_transform.yaml"),
        max_particles=53,
        train_type="event",
        sin_cos=True,
        zero_neutral_vtx=False,
        mode="eval",
        reduce_ds=10,  # more kept events than the file has
        eta_clamp=(-3.0, 5.0),
        class_based_scaling=True,
        event_selector=HasElectron(),
    )
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first["ctxt_mask"].sum() == 4 and second["ctxt_mask"].sum() == 3
    assert first["ctxt_data"].shape[0] == 53
    assert [int(first["eventNumber"]), int(second["eventNumber"])] == [0, 2]
    assert dataset._event_data["truth_Ng"].tolist() == [1, 2]
    assert dataset._event_data["truth_Nn"].tolist() == [1, 0]
    # Particles are ordered by p, highest first.
    collection_data, _, _, _ = dataset._get_collections(0)
    p = collection_data["truth"]["p"][:4].tolist()
    assert p == sorted(p, reverse=True)
