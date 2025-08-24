import torch
import torch.nn as nn
import pytest

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.sampling import EulerSampler, PNDMSampler, DPM_Solver

torch.manual_seed(42)

FS_DIM = 4


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(FS_DIM, FS_DIM)

    def forward(self, fs_data, timestep, ctxt_dict):
        return self.linear(fs_data) + timestep.mean()


class MockFlowMatcher:
    def __init__(self, flow_type="cfm"):
        self.flow_type = flow_type

    def sample(self, pf_data):
        batch_size = pf_data.shape[0]
        t = torch.rand((batch_size,), dtype=pf_data.dtype, device=pf_data.device)
        xt = pf_data + 0.1 * torch.randn_like(pf_data)
        ut = 0.1 * torch.randn_like(pf_data)
        return t, xt, ut


class MockLoss:
    def __call__(self, vt, ut, fs_mask=None):
        if fs_mask is not None:
            vt = vt[fs_mask]
            ut = ut[fs_mask]
        return nn.functional.mse_loss(vt, ut)


class TestFlowLightning:
    COSINE_SCHEDULER_DEFAULTS = {
        "lr_scheduler_type": "cosine",
        "first_cycle_steps": 10,
        "cycle_mult": 1.0,
        "max_lr": 0.001,
        "min_lr": 1e-5,
        "warmup_steps": 0,
        "gamma": 1.0,
        "skip_scheduler": False,
    }
    WARMUP_DECAY_SCHEDULER_DEFAULTS = {
        "lr_scheduler_type": "warmup_decay",
        "warmup_steps": 10,
        "decay_steps": 50,
        "min_lr": 1e-5,
        "skip_scheduler": False,
    }

    @pytest.fixture(params=["cfm", "rcfm"])
    def default_config(self, request):
        flow_type = request.param
        model = MockModel()
        fm = MockFlowMatcher(flow_type=flow_type)
        loss_fn = MockLoss()
        return {
            "name": "test",
            "model": model,
            "fm": fm,
            "loss": loss_fn,
            "n_steps": 10,
            "sampler_type": "euler",
            "optimizer": "AdamW",
            "optimizer_kwargs": {"lr": 1e-3},
            "lrs_config": {"skip_scheduler": True},
        }

    @pytest.fixture
    def flow_lightning(self, default_config):
        return FlowLightning(**default_config)

    @pytest.fixture(params=[(2, FS_DIM), (2, 5, FS_DIM)])
    def data(self, request):
        fs_shape = request.param
        fs_data = torch.randn(fs_shape, dtype=torch.float32)
        ctxt_data = torch.randn((2, 5, 4))
        fs_mask = (
            torch.ones(fs_shape[:2], dtype=torch.bool) if len(fs_shape) == 3 else None
        )
        ctxt_dict = {"fs_mask": fs_mask, "ctxt_data": ctxt_data}
        return fs_data, ctxt_dict

    def test_init(self, flow_lightning, default_config):
        assert flow_lightning.name == default_config["name"]
        assert isinstance(flow_lightning.sampler, EulerSampler)
        assert flow_lightning.sampler.reverse_time == (
            flow_lightning.FM.flow_type == "rcfm"
        )
        assert flow_lightning.sampler.n_steps == default_config["n_steps"]

    def test_forward(self, data, flow_lightning):
        output = flow_lightning(data)
        assert output.ndim == 0

    @pytest.mark.parametrize("n_steps", [5])
    @pytest.mark.parametrize("sampler_type", ["euler", "pndm", "dpm"])
    def test_change_sampler(self, flow_lightning, n_steps, sampler_type):
        flow_lightning.change_sampler(sampler_type, n_steps)
        assert flow_lightning.n_steps == n_steps

        match sampler_type:
            case "euler":
                assert isinstance(flow_lightning.sampler, EulerSampler)
                assert flow_lightning.sampler.reverse_time == (
                    flow_lightning.FM.flow_type == "rcfm"
                )
                assert flow_lightning.sampler.n_steps == n_steps
            case "pndm":
                assert isinstance(flow_lightning.sampler, PNDMSampler)
                assert flow_lightning.sampler.reverse_time == (
                    flow_lightning.FM.flow_type == "rcfm"
                )
                assert flow_lightning.sampler.n_steps == n_steps
            case "dpm":
                assert isinstance(flow_lightning.sampler, DPM_Solver)

    @pytest.mark.parametrize("n_steps", [5])
    @pytest.mark.parametrize("data", [(2, FS_DIM), (2, 5, FS_DIM)], indirect=True)
    @pytest.mark.parametrize("sampler_type", ["euler", "pndm", "dpm"])
    def test_sample(self, flow_lightning, n_steps, data, sampler_type):
        fs_data, ctxt_dict = data
        fs_shape = fs_data.shape
        flow_lightning.change_sampler(sampler_type, n_steps)
        flow_lightning.net.time_shape = (
            (fs_shape[0], 1, 1) if len(fs_shape) == 3 else (fs_shape[0], 1)
        )
        out = flow_lightning.sample(fs_shape, ctxt_dict)
        assert out.shape == fs_shape

    @pytest.mark.parametrize("optimizer", ["AdamW", "Lion"])
    def test_configure_optimizers(self, flow_lightning, optimizer):
        flow_lightning.optimizer = optimizer.lower()
        optimizers = flow_lightning.configure_optimizers()
        assert isinstance(optimizers, torch.optim.Optimizer)
        assert optimizers.param_groups[0]["lr"] == flow_lightning.optimizer_kwargs["lr"]

    @pytest.mark.parametrize(
        "lrs_config",
        [COSINE_SCHEDULER_DEFAULTS, WARMUP_DECAY_SCHEDULER_DEFAULTS],
        ids=["cosine", "warmup_decay"],
    )
    @pytest.mark.parametrize("interval", ["step", "epoch"])
    def test_configure_optimizers_with_scheduler(
        self, flow_lightning, lrs_config, interval
    ):
        flow_lightning.lrs_config = lrs_config
        flow_lightning.lrs_config["interval"] = interval
        optimizers = flow_lightning.configure_optimizers()
        if lrs_config.get("skip_scheduler", False):
            assert isinstance(optimizers, torch.optim.Optimizer)
        else:
            assert isinstance(optimizers, dict)
            assert isinstance(optimizers["optimizer"], torch.optim.Optimizer)
            assert isinstance(optimizers["lr_scheduler"], dict)
            assert isinstance(
                optimizers["lr_scheduler"]["scheduler"],
                torch.optim.lr_scheduler._LRScheduler,
            )
            assert optimizers["lr_scheduler"]["interval"] == interval
