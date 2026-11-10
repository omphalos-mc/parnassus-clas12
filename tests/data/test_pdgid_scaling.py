import pytest
import torch
import yaml

from parnassus_core.data.physics_data import build_ctxt_data, build_fs_data
from parnassus_core.data.utils.scaling import FeatureScaler, Unscaler
from parnassus_core.data.utils.variable_helper import VariableHelper

SCALE_CONFIG = {
    "p": {"mean": 1.6, "std": 1.7},
    "theta": {"mean": 0.4, "std": 0.3},
    "pdgId_22": {"p": {"mean": 0.6, "std": 0.7}, "theta": {"mean": 0.5, "std": 0.4}},
    "pdgId_11": {"p": {"mean": 4.1, "std": 2.0}, "theta": {"mean": 0.3, "std": 0.1}},
    "pdgId_-211": {"p": {"mean": 1.1, "std": 1.0}, "theta": {"mean": 0.5, "std": 0.4}},
    "pdgId_0": {"p": {"mean": 2.0, "std": 1.2}, "theta": {"mean": 0.3, "std": 0.2}},
}
# pdgId -> block; K0L (130) has no one-hot bucket, so it uses pdgId_0.
BLOCK = {11: 11, 22: 22, -211: -211, 130: 0}
FS_VARS = ["pflow_p", "pflow_theta", "pflow_pdgId"]
CTXT_VARS = ["truth_p", "truth_theta", "truth_pdgId"]


def _scaler(tmp_path, config=SCALE_CONFIG, class_based_scaling=True):
    path = tmp_path / "scale.yaml"
    path.write_text(yaml.safe_dump(config))
    return FeatureScaler(str(path), class_based_scaling=class_based_scaling)


@pytest.fixture
def var_helper():
    return VariableHelper(["truth", "pflow"])


@pytest.fixture
def batch():
    """Two events of four slots, the same particles in truth and pflow."""
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    coll = {
        "p": torch.tensor([[5.0, 0.3, 1.2, 2.5], [0.8, 4.4, 0.0, 0.0]]),
        "theta": torch.tensor([[0.2, 0.9, 0.5, 0.3], [0.6, 0.25, 0.0, 0.0]]),
        "pdgId": torch.tensor([[11.0, 22.0, -211.0, 130.0], [22.0, 11.0, 0.0, 0.0]]),
    }
    data = {"truth": coll, "pflow": {k: v.clone() for k, v in coll.items()}}
    return data, {"truth": mask, "pflow": mask.clone()}


def _fs(batch, scaler, var_helper):
    data, masks = batch
    fs_data, _ = build_fs_data(
        FS_VARS,
        data,
        masks,
        {},
        sin_cos=False,
        feature_scaler=scaler,
        var_helper=var_helper,
        train_type="particle",
    )
    return fs_data


def test_pdgid_blocks_parse(tmp_path):
    scaler = _scaler(tmp_path)
    assert scaler.group_var == "pdgId"
    assert set(scaler.class_transforms) == {22, 11, -211, 0}


def test_class_and_pdgid_blocks_cannot_mix(tmp_path):
    config = SCALE_CONFIG | {"class_0": {"p": {"mean": 1.0, "std": 1.0}}}
    with pytest.raises(ValueError, match="cannot be mixed"):
        _scaler(tmp_path, config)


def test_forward_scaling_uses_pdgid_block(tmp_path, batch, var_helper):
    data, masks = batch
    fs_data = _fs(batch, _scaler(tmp_path), var_helper)

    for event, slot in masks["pflow"].nonzero().tolist():
        block = SCALE_CONFIG[f"pdgId_{BLOCK[int(data['pflow']['pdgId'][event, slot])]}"]
        for col, var in enumerate(("p", "theta")):
            raw = data["pflow"][var][event, slot].item()
            expected = (raw - block[var]["mean"]) / block[var]["std"]
            assert fs_data[event, slot, col].item() == pytest.approx(expected, abs=1e-6)


def test_unscaler_round_trip(tmp_path, batch, var_helper):
    data, masks = batch
    scaler = _scaler(tmp_path)
    fs_data = _fs(batch, scaler, var_helper)
    ctxt_dict = build_ctxt_data(
        CTXT_VARS, {"truth": data["truth"]}, masks, False, scaler, var_helper
    )
    unscaler = Unscaler(scaler, FS_VARS, CTXT_VARS, [], sin_cos=False)

    out = unscaler.unscale_variables(
        fs_data, ctxt_dict, target_data=fs_data, unshift=False
    )

    for prefix, coll in (("fs", "pflow"), ("pflow", "pflow"), ("truth", "truth")):
        valid = masks[coll]
        for var in ("p", "theta"):
            torch.testing.assert_close(
                torch.from_numpy(out[f"{prefix}_{var}"])[valid],
                data[coll][var][valid],
            )


def test_pdgid_blocks_ignored_without_class_based_scaling(tmp_path, batch, var_helper):
    data, masks = batch
    scaler = _scaler(tmp_path, class_based_scaling=False)
    assert scaler.group_var is None

    fs_data = _fs(batch, scaler, var_helper)

    valid = masks["pflow"]
    torch.testing.assert_close(
        fs_data[..., 0][valid], ((data["pflow"]["p"] - 1.6) / 1.7)[valid]
    )
