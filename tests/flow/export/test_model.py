import torch
import torch.nn as nn
import pytest

from parnassus_core.flow.export.model import ExportModelWrapper
from parnassus_core.flow.export.utils import (
    EVENT_TORCH_DYNAMIC_AXES,
    PARTICLE_TORCH_DYNAMIC_AXES,
)

torch.manual_seed(42)

HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"

BS = 2
SEQ_LEN = 8
FS_DIM = 4
CTXT_DIM = 6
GLOBAL_DIM = 3


class MockNet(nn.Module):
    """Mock network for testing."""

    def __init__(self, fs_dim=FS_DIM, output_dim=FS_DIM):
        super().__init__()
        self.linear = nn.Linear(fs_dim, output_dim)

    def forward(self, fs_data, timestep, ctxt_dict):
        # Simple mock forward pass
        output = self.linear(fs_data)
        # Add timestep influence
        output = output + timestep.mean() * 0.1
        if fs_data.dim() == 3:
            output = output * ctxt_dict["fs_mask"].unsqueeze(-1)
        return output


@pytest.fixture(params=["event", "particle"])
def model_type(request):
    return request.param


@pytest.fixture
def mock_net():
    return MockNet()


@pytest.fixture
def input_data_event():
    """Create event-type input data."""
    fs_data = torch.randn(BS, FS_DIM, device=DEVICE)
    timestep = torch.randn(BS, 1, device=DEVICE)
    mask = torch.ones(BS, SEQ_LEN, device=DEVICE).bool()
    ctxt_data = torch.randn(BS, SEQ_LEN, CTXT_DIM, device=DEVICE)
    ctxt_global_data = torch.randn(BS, GLOBAL_DIM, device=DEVICE)

    return fs_data, timestep, mask, ctxt_data, ctxt_global_data


@pytest.fixture
def input_data_particle():
    """Create particle-type input data."""
    fs_data = torch.randn(BS, SEQ_LEN, FS_DIM, device=DEVICE)
    timestep = torch.randn(BS, 1, device=DEVICE)
    # For particle type, mask has an additional dimension
    mask = torch.ones(BS, SEQ_LEN, 2, device=DEVICE).bool()
    mask[:, SEQ_LEN // 2 :, 0] = False  # fs_mask
    mask[:, SEQ_LEN // 2 :, 1] = False  # ctxt_mask
    ctxt_data = torch.randn(BS, SEQ_LEN, CTXT_DIM, device=DEVICE)
    ctxt_global_data = torch.randn(BS, GLOBAL_DIM, device=DEVICE)

    return fs_data, timestep, mask, ctxt_data, ctxt_global_data


class TestExportModelWrapper:
    @pytest.mark.parametrize(
        "model_type, dynamic_axes",
        [
            ("event", EVENT_TORCH_DYNAMIC_AXES),
            ("particle", PARTICLE_TORCH_DYNAMIC_AXES),
        ],
        ids=["event", "particle"],
    )
    def test_init(self, mock_net, model_type, dynamic_axes):
        """Test initialization for event model type."""
        wrapper = ExportModelWrapper(
            net=mock_net, dynamic_axes=dynamic_axes, model_type=model_type
        )

        assert wrapper.net is mock_net
        assert wrapper.dynamic_axes == dynamic_axes
        assert wrapper.model_type == model_type

    def test_forward_event(self, mock_net, input_data_event):
        """Test forward pass for event model type."""
        wrapper = ExportModelWrapper(
            net=mock_net, dynamic_axes=EVENT_TORCH_DYNAMIC_AXES, model_type="event"
        ).to(DEVICE)

        fs_data, timestep, mask, ctxt_data, ctxt_global_data = input_data_event

        with torch.no_grad():
            output = wrapper(fs_data, timestep, mask, ctxt_data, ctxt_global_data)

        assert output.shape == fs_data.shape
        assert output.device == fs_data.device
        assert output.dtype == fs_data.dtype

    def test_forward_particle(self, mock_net, input_data_particle):
        """Test forward pass for event model type."""
        wrapper = ExportModelWrapper(
            net=mock_net,
            dynamic_axes=PARTICLE_TORCH_DYNAMIC_AXES,
            model_type="particle",
        ).to(DEVICE)

        fs_data, timestep, mask, ctxt_data, ctxt_global_data = input_data_particle

        with torch.no_grad():
            output = wrapper(fs_data, timestep, mask, ctxt_data, ctxt_global_data)

        assert output.shape == fs_data.shape
        assert output.device == fs_data.device
        assert output.dtype == fs_data.dtype
        assert torch.all(output[~mask[..., 1]] == 0)
