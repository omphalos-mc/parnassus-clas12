from unittest.mock import Mock

import pytest
import torch
from torch import nn

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.modules.optimizer import Lion
from parnassus_core.flow.sampling import (
    DPM_Solver,
    EulerSampler,
    PNDMSampler,
    SDESampler,
)
from parnassus_core.flow.utils.flow_matcher import FlowMatcher, FlowSample
from parnassus_core.flow.utils.scheduler import (
    CosineAnnealingWarmupRestarts,
    WarmupDecayConstLR,
)


FS_DIM = 4


class MockModel(nn.Module):
    def __init__(self, prediction=0.0, guidance=None):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(FS_DIM))
        self.bias = nn.Parameter(torch.zeros(FS_DIM))
        self.prediction = prediction
        self.guidance = guidance

    def forward(self, fs_data, timestep, ctxt_dict):
        del timestep, ctxt_dict
        prediction = torch.zeros_like(fs_data) + self.prediction
        guidance = (
            None if self.guidance is None else torch.zeros_like(fs_data) + self.guidance
        )
        return prediction, guidance


class RecordingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = None
        self.target = None
        self.mask = None

    def forward(self, prediction, target, mask=None):
        self.prediction = prediction
        self.target = target
        self.mask = mask
        return nn.functional.mse_loss(
            prediction[mask] if mask is not None else prediction,
            target[mask] if mask is not None else target,
        )


def make_flow(**overrides):
    config = {
        "name": "test",
        "model": MockModel(),
        "fm": FlowMatcher(flow_type="icfm"),
        "loss": RecordingLoss(),
        "n_steps": 10,
        "sampler_type": "euler",
        "optimizer": "AdamW",
        "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 0.1},
        "lrs_config": {"skip_scheduler": True},
    }
    config.update(overrides)
    return FlowLightning(**config)


def fixed_sample(data, *, source=2.0, time=0.25):
    source_tensor = torch.full_like(data, source)
    data_tensor = data.clone()
    time_tensor = torch.full(
        (data.shape[0],),
        time,
        dtype=data.dtype,
        device=data.device,
    )
    path = FlowMatcher("icfm").path
    return FlowSample(
        time=time_tensor,
        state=path.interpolate(source_tensor, data_tensor, time_tensor),
        velocity=path.velocity(source_tensor, data_tensor, time_tensor),
        source=source_tensor,
        data=data_tensor,
    )


@pytest.mark.parametrize("flow_type", ["icfm", "rcfm", "vp"])
def test_initialization_uses_requested_flow_and_defaults(flow_type):
    flow = make_flow(
        fm=FlowMatcher(flow_type=flow_type),
        n_steps=6,
    )

    assert flow.name == "test"
    assert flow.loss_target_type == flow.prediction_type
    assert isinstance(flow.sampler, EulerSampler)
    assert flow.sampler.reverse_time is (flow_type == "rcfm")
    assert flow.sampler.n_steps == 6


def test_forward_converts_prediction_to_loss_target_and_passes_particle_mask(
    monkeypatch,
):
    data = torch.full((2, 3, FS_DIM), 6.0)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    loss_fn = RecordingLoss()
    flow = make_flow(
        model=MockModel(prediction=6.0),
        loss=loss_fn,
        prediction_type="data",
        loss_target_type="noise",
    )
    sample = fixed_sample(data)
    monkeypatch.setattr(flow.FM, "sample", lambda _: sample)

    loss = flow((data, {"fs_mask": mask, "ctxt_data": data}))

    torch.testing.assert_close(loss_fn.prediction, sample.source)
    assert loss_fn.target is sample.source
    assert loss_fn.mask is mask
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_forward_does_not_pass_sequence_mask_for_event_data(monkeypatch):
    data = torch.zeros(2, FS_DIM)
    loss_fn = RecordingLoss()
    flow = make_flow(loss=loss_fn)
    monkeypatch.setattr(flow.FM, "sample", lambda _: fixed_sample(data))

    flow((data, {"fs_mask": None, "ctxt_data": data}))

    assert loss_fn.mask is None


def test_internal_guidance_loss_is_added_with_configured_weight(monkeypatch):
    data = torch.zeros(2, FS_DIM)
    flow = make_flow(
        model=MockModel(prediction=0.0, guidance=1.0),
        ig_loss_weight=0.25,
    )
    sample = fixed_sample(data, source=0.0)
    monkeypatch.setattr(flow.FM, "sample", lambda _: sample)

    loss = flow((data, {"fs_mask": None, "ctxt_data": data}))

    torch.testing.assert_close(loss, torch.tensor(0.25))


@pytest.mark.parametrize("shape", [(2, FS_DIM), (2, 5, FS_DIM)])
@pytest.mark.parametrize("prediction_type", ["velocity", "data", "noise"])
@pytest.mark.parametrize("loss_target_type", ["velocity", "data", "noise"])
def test_all_prediction_and_loss_target_combinations_work(
    shape,
    prediction_type,
    loss_target_type,
):
    data = torch.randn(shape)
    fs_mask = torch.ones(shape[:2], dtype=torch.bool) if len(shape) == 3 else None
    flow = make_flow(
        fm=FlowMatcher(flow_type="icfm"),
        prediction_type=prediction_type,
        loss_target_type=loss_target_type,
    )

    loss = flow(
        (
            data,
            {
                "fs_mask": fs_mask,
                "ctxt_data": torch.randn(2, 5, FS_DIM),
            },
        )
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert flow.loss_target_type == loss_target_type


@pytest.mark.parametrize(
    ("time", "weight", "expected"),
    [
        (0.5, 2.0, 7.0),
        (0.1, 2.0, 4.0),
        (0.5, 1.0, 4.0),
    ],
)
def test_sampling_guidance_is_applied_only_inside_window(time, weight, expected):
    flow = make_flow(model=MockModel(prediction=4.0, guidance=1.0))
    flow.change_sampler(
        "euler",
        ig_sampling_weight=weight,
        ig_t_min=0.25,
        ig_t_max=0.75,
    )
    state = torch.zeros(2, FS_DIM)

    output = flow.sampler.model_fn(
        state,
        torch.full((state.shape[0],), time),
        {"ctxt_data": state},
    )

    torch.testing.assert_close(output, torch.full_like(state, expected))


@pytest.mark.parametrize(
    ("sampler_type", "expected_type"),
    [
        ("euler", EulerSampler),
        ("sde", SDESampler),
        ("pndm", PNDMSampler),
        ("dpm", DPM_Solver),
    ],
)
def test_change_sampler_selects_implementation(sampler_type, expected_type):
    flow = make_flow(fm=FlowMatcher(flow_type="rcfm"))

    flow.change_sampler(sampler_type, n_steps=7)

    assert isinstance(flow.sampler, expected_type)
    assert flow.n_steps == 7
    if hasattr(flow.sampler, "n_steps"):
        assert flow.sampler.n_steps == 7


@pytest.mark.parametrize(
    ("flow_type", "sampler_type"),
    [
        ("icfm", "euler"),
        ("rcfm", "sde"),
        ("vp", "pndm"),
        ("icfm", "dpm"),
    ],
)
@pytest.mark.parametrize("prediction_type", ["velocity", "data", "noise"])
def test_all_supported_samplers_work_with_every_prediction_type(
    flow_type,
    sampler_type,
    prediction_type,
):
    shape = (2, FS_DIM)
    flow = make_flow(
        fm=FlowMatcher(flow_type=flow_type),
        prediction_type=prediction_type,
    )
    flow.change_sampler(
        sampler_type,
        n_steps=4,
        sde_eta=0.0,
        sde_last_step_size=1e-3,
    )

    result = flow.sample(
        shape,
        {
            "ctxt_data": torch.randn(shape),
        },
    )

    assert result.shape == shape
    assert torch.isfinite(result).all()


def test_sample_delegates_runtime_options():
    flow = make_flow()
    flow.sampler = Mock()
    flow.sampler.sample.return_value = torch.ones(2, FS_DIM)
    ctxt_dict = {"ctxt_data": torch.zeros(2, 1)}

    result = flow.sample((2, FS_DIM), ctxt_dict, n_steps=3, to_cpu=False)

    torch.testing.assert_close(result, torch.ones(2, FS_DIM))
    flow.sampler.sample.assert_called_once_with(
        target_shape=(2, FS_DIM),
        ctxt_dict=ctxt_dict,
        to_cpu=False,
        n_steps=3,
    )


@pytest.mark.parametrize(
    ("optimizer_name", "optimizer_type"),
    [("AdamW", torch.optim.AdamW), ("Lion", Lion)],
)
def test_configure_optimizers_excludes_bias_from_weight_decay(
    optimizer_name, optimizer_type
):
    flow = make_flow(optimizer=optimizer_name)

    optimizer = flow.configure_optimizers()

    assert isinstance(optimizer, optimizer_type)
    assert optimizer.param_groups[0]["params"] == [flow.net.weight]
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["params"] == [flow.net.bias]
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


@pytest.mark.parametrize(
    ("scheduler_type", "scheduler_kwargs", "expected_type"),
    [
        (
            "cosine",
            {
                "first_cycle_steps": 10,
                "warmup_steps": 2,
                "max_lr": 1e-3,
                "min_lr": 1e-5,
            },
            CosineAnnealingWarmupRestarts,
        ),
        (
            "warmup_decay",
            {"warmup_steps": 2, "decay_steps": 5, "min_lr": 1e-5},
            WarmupDecayConstLR,
        ),
    ],
)
def test_configure_optimizers_builds_requested_scheduler(
    scheduler_type, scheduler_kwargs, expected_type
):
    flow = make_flow(
        lrs_config={
            "skip_scheduler": False,
            "lr_scheduler_type": scheduler_type,
            "lr_scheduler_kwargs": scheduler_kwargs,
            "interval": "step",
        }
    )

    optimizers, schedulers = flow.configure_optimizers()

    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert isinstance(schedulers[0]["scheduler"], expected_type)
    assert schedulers[0]["interval"] == "step"


def test_configure_optimizers_builds_onecycle_scheduler():
    flow = make_flow(
        lrs_config={
            "skip_scheduler": False,
            "lr_scheduler_type": "onecycle",
            "lr_scheduler_kwargs": {
                "max": 1e-2,
                "initial": 1e-3,
                "end": 1e-5,
                "pct_start": 0.25,
            },
            "interval": "step",
        }
    )
    flow._trainer = Mock(estimated_stepping_batches=8)

    optimizers, schedulers = flow.configure_optimizers()

    assert len(optimizers) == 1
    assert isinstance(schedulers[0]["scheduler"], torch.optim.lr_scheduler.OneCycleLR)
    assert schedulers[0]["interval"] == "step"


def test_configure_optimizers_rejects_unknown_optimizer():
    flow = make_flow(optimizer="not-an-optimizer")

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        flow.configure_optimizers()


def test_configure_optimizers_rejects_unknown_scheduler():
    flow = make_flow(
        lrs_config={
            "skip_scheduler": False,
            "lr_scheduler_type": "not-a-scheduler",
        }
    )

    with pytest.raises(ValueError, match="Unsupported lr_scheduler_type"):
        flow.configure_optimizers()
