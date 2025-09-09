import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from torch.utils.data import Dataset
import tempfile

from parnassus_core.flow.export.utils import (
    load_model_and_config,
    export,
    validate,
    export_and_validate,
    EVENT_ONNX_DYNAMIC_AXES,
    PARTICLE_ONNX_DYNAMIC_AXES,
    EVENT_TORCH_DYNAMIC_AXES,
    PARTICLE_TORCH_DYNAMIC_AXES,
)
from parnassus_core.flow.export.model import ExportModelWrapper

torch.manual_seed(42)

HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"

BS = 2
SEQ_LEN = 8
FS_DIM = 4
CTXT_DIM = 6
GLOBAL_DIM = 3


class MockOnnxNode:
    def __init__(self, name):
        self.name = name


class MockNet(nn.Module):
    """Mock network for testing."""

    def __init__(self, fs_dim=FS_DIM, output_dim=FS_DIM, ctxt_dim=CTXT_DIM):
        super().__init__()
        self.linear = nn.Linear(fs_dim, output_dim)
        self.ctxt_linear = nn.Linear(ctxt_dim, output_dim)

    def change_backend(self, backend):
        pass  # Mock method

    def forward(self, fs_data, timestep, ctxt_dict):
        ctxt_embd = self.ctxt_linear(ctxt_dict["ctxt_data"])
        output = self.linear(fs_data) + (
            ctxt_embd * ctxt_dict["ctxt_mask"].unsqueeze(-1)
        ).mean(1)
        output = output + timestep.mean() * 0.1
        if fs_data.dim() == 3:
            output = output * ctxt_dict["fs_mask"].unsqueeze(-1)
        return output


class MockFlowLightning:
    """Mock FlowLightning model for testing."""

    def __init__(self):
        self.net = MockNet()

    def eval(self):
        self.net.eval()
        return self

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, map_location="cpu"):
        return cls()


class MockDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, train_type="event", length=10, impact=False):
        self.train_type = train_type
        self.length = length
        self.impact = impact

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fs_data = torch.randn(SEQ_LEN, FS_DIM)
        if self.train_type == "particle":
            ctxt_dict = {
                "ctxt_mask": torch.ones(SEQ_LEN).bool(),
                "fs_mask": torch.ones(SEQ_LEN).bool(),
                "ctxt_data": torch.randn(SEQ_LEN, CTXT_DIM),
                "ctxt_global_data": torch.randn(GLOBAL_DIM),
            }
            if self.impact:
                ctxt_dict["pf_ctxt_data"] = torch.randn(
                    SEQ_LEN, 5
                )  # Example PF ctxt data
            # Set half the masks to False
            ctxt_dict["ctxt_mask"][SEQ_LEN // 2 :] = False
            ctxt_dict["fs_mask"][SEQ_LEN // 2 :] = False
        else:
            ctxt_dict = {
                "ctxt_mask": torch.ones(SEQ_LEN).bool(),
                "ctxt_data": torch.randn(SEQ_LEN, CTXT_DIM),
                "ctxt_global_data": torch.randn(GLOBAL_DIM),
            }
            ctxt_dict["ctxt_mask"][SEQ_LEN // 2 :] = False

        return fs_data, ctxt_dict


def get_config(train_type="event"):
    return {"data": {"train_type": train_type}}


@pytest.fixture
def mock_config():
    return get_config("event")


@pytest.fixture
def mock_config_particle():
    return get_config("particle")


def get_sample_batch(train_type="event", impact=False):
    if train_type == "event":
        output = {
            "fs_data": torch.randn(BS, FS_DIM),
            "timestep": torch.randn(BS, 1),
            "mask": torch.ones(BS, SEQ_LEN).bool(),
            "ctxt_data": torch.randn(BS, SEQ_LEN, CTXT_DIM),
            "ctxt_global_data": torch.randn(BS, GLOBAL_DIM),
        }
    else:  # particle
        mask = torch.ones(BS, SEQ_LEN, 2).bool()
        mask[:, SEQ_LEN // 2 :, :] = False
        output = {
            "fs_data": torch.randn(BS, SEQ_LEN, FS_DIM),
            "timestep": torch.randn(BS, 1),
            "mask": mask,
            "ctxt_data": torch.randn(BS, SEQ_LEN, CTXT_DIM),
            "ctxt_global_data": torch.randn(BS, GLOBAL_DIM),
        }
        if impact:
            output["pf_ctxt_data"] = torch.randn(BS, SEQ_LEN, 5)
    return output


@pytest.fixture
def sample_batch_event():
    return get_sample_batch("event")


class TestDynamicAxes:
    """Test the predefined dynamic axes constants."""

    @pytest.mark.parametrize(
        "dynamic_axes",
        [
            EVENT_ONNX_DYNAMIC_AXES,
            PARTICLE_ONNX_DYNAMIC_AXES,
        ],
        ids=["event", "particle"],
    )
    def test_onnx_dynamic_axes(self, dynamic_axes):
        expected_keys = {"fs_data", "timestep", "mask", "ctxt_data", "ctxt_global_data"}
        assert set(dynamic_axes.keys()) == expected_keys

        # Check that batch dimension (0) is dynamic for all tensors
        for key, axes in dynamic_axes.items():
            assert 0 in axes
            assert axes[0] == "batch"

    @pytest.mark.parametrize(
        "dynamic_axes",
        [
            EVENT_TORCH_DYNAMIC_AXES,
            PARTICLE_TORCH_DYNAMIC_AXES,
        ],
        ids=["event", "particle"],
    )
    def test_event_torch_dynamic_axes(self, dynamic_axes):
        expected_keys = {"fs_data", "timestep", "mask", "ctxt_data", "ctxt_global_data"}
        assert set(dynamic_axes.keys()) == expected_keys

        # Check that batch dimension (0) is dynamic for all tensors
        for key, axes in dynamic_axes.items():
            assert 0 in axes
            assert hasattr(axes[0], "__name__")  # Should be a Dim object with name


class TestLoadModelAndConfig:
    """Test model and configuration loading."""

    @patch("builtins.open", mock_open(read_data="data:\n  train_type: event\n"))
    @patch("parnassus_core.flow.export.utils.FlowLightning")
    def test_load_model_and_config(self, mock_flow_lightning):
        mock_model = MockFlowLightning()
        mock_flow_lightning.load_from_checkpoint.return_value = mock_model

        model, config = load_model_and_config("config.yaml", "model.ckpt")

        assert model == mock_model
        assert config == {"data": {"train_type": "event"}}
        mock_flow_lightning.load_from_checkpoint.assert_called_once_with(
            "model.ckpt", map_location="cpu"
        )


class TestExport:
    """Test model export functionality."""

    @pytest.mark.parametrize(
        "model_type, impact",
        [("event", False), ("particle", False), ("particle", True)],
        ids=["event", "particle", "particle_impact"],
    )
    def test_export_torch(self, model_type, impact):
        mock_model = MockFlowLightning()
        mock_config = get_config(model_type)
        sample_batch = get_sample_batch(model_type, impact=impact)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.pt"

            with (
                patch("torch.export.export") as mock_export,
                patch("torch.export.save") as mock_save,
            ):
                mock_exported_model = Mock()
                mock_export.return_value = mock_exported_model

                model_wrapper, result_path = export(
                    mock_model, mock_config, "torch", output_path, sample_batch
                )

                assert isinstance(model_wrapper, ExportModelWrapper)
                assert result_path == output_path
                assert model_wrapper.model_type == model_type

                mock_export.assert_called_once()
                mock_save.assert_called_once_with(mock_exported_model, output_path)

    @pytest.mark.parametrize(
        "model_type, impact",
        [("event", False), ("particle", False), ("particle", True)],
        ids=["event", "particle", "particle_impact"],
    )
    @patch("torch.onnx.export")
    def test_export_onnx(self, mock_onnx_export, model_type, impact):
        mock_model = MockFlowLightning()
        mock_config = get_config(model_type)
        sample_batch = get_sample_batch(model_type, impact=impact)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.onnx"

            model_wrapper, result_path = export(
                mock_model, mock_config, "onnx", output_path, sample_batch
            )

            assert isinstance(model_wrapper, ExportModelWrapper)
            assert result_path == output_path
            assert model_wrapper.model_type == model_type

            mock_onnx_export.assert_called_once()
            args, kwargs = mock_onnx_export.call_args
            assert kwargs["f"] == output_path
            assert kwargs["opset_version"] == 16
            assert "input_names" in kwargs
            assert "dynamic_axes" in kwargs

    def test_export_unknown_mode(self, mock_config, sample_batch_event):
        mock_model = MockFlowLightning()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.unknown"

            with pytest.raises(ValueError, match="Unknown mode"):
                export(
                    mock_model, mock_config, "unknown", output_path, sample_batch_event
                )

    def test_export_unknown_data_type(self, sample_batch_event):
        mock_model = MockFlowLightning()
        config = {"data": {"train_type": "unknown"}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.pt"

            with pytest.raises(
                ValueError, match=rf"Unknown data type: {config['data']['train_type']}"
            ):
                export(mock_model, config, "torch", output_path, sample_batch_event)


class TestValidate:
    """Test model validation functionality."""

    def test_validate_torch_single_batch(self, sample_batch_event):
        mock_net = MockNet()
        model_wrapper = ExportModelWrapper(
            net=mock_net, dynamic_axes=EVENT_TORCH_DYNAMIC_AXES, model_type="event"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.pt"

            # Mock the torch.export.load to return a model that gives same output
            mock_loaded_model = Mock()
            mock_loaded_model.return_value = model_wrapper(
                sample_batch_event["fs_data"],
                sample_batch_event["timestep"],
                sample_batch_event["mask"],
                sample_batch_event["ctxt_data"],
                sample_batch_event["ctxt_global_data"],
            )

            mock_exported_program = Mock()
            mock_exported_program.module.return_value = mock_loaded_model

            with (
                patch("torch.export.load", return_value=mock_exported_program),
                patch("torch.compile", return_value=mock_loaded_model),
            ):
                is_valid = validate(
                    model_wrapper, output_path, sample_batch_event, mode="torch"
                )

                assert is_valid

    def test_validate_torch_multiple_batches(self, sample_batch_event):
        mock_net = MockNet()
        model_wrapper = ExportModelWrapper(
            net=mock_net, dynamic_axes=EVENT_TORCH_DYNAMIC_AXES, model_type="event"
        )

        # Create multiple batches
        batch2 = {k: v.clone() for k, v in sample_batch_event.items()}
        batches = [sample_batch_event, batch2]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.pt"

            mock_loaded_model = Mock()
            # Mock to return consistent outputs
            mock_loaded_model.side_effect = [
                model_wrapper(
                    batch["fs_data"],
                    batch["timestep"],
                    batch["mask"],
                    batch["ctxt_data"],
                    batch["ctxt_global_data"],
                )
                for batch in batches
            ]

            mock_exported_program = Mock()
            mock_exported_program.module.return_value = mock_loaded_model

            with (
                patch("torch.export.load", return_value=mock_exported_program),
                patch("torch.compile", return_value=mock_loaded_model),
            ):
                is_valid = validate(model_wrapper, output_path, batches, mode="torch")

                assert is_valid

    @patch("onnx.load")
    @patch("onnx.checker.check_model")
    @patch("onnxruntime.InferenceSession")
    def test_validate_onnx(
        self, mock_ort_session, mock_check_model, mock_load, sample_batch_event
    ):
        mock_net = MockNet()
        model_wrapper = ExportModelWrapper(
            net=mock_net, dynamic_axes=EVENT_ONNX_DYNAMIC_AXES, model_type="event"
        )

        # Mock ONNX runtime session
        expected_output = model_wrapper(
            sample_batch_event["fs_data"],
            sample_batch_event["timestep"],
            sample_batch_event["mask"],
            sample_batch_event["ctxt_data"],
            sample_batch_event["ctxt_global_data"],
        )

        mock_session = Mock()
        mock_session.get_inputs.return_value = [
            MockOnnxNode(name) for name in sample_batch_event.keys()
        ]
        mock_session.run.return_value = [expected_output.numpy()]
        mock_ort_session.return_value = mock_session

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.onnx"

            is_valid = validate(
                model_wrapper, output_path, sample_batch_event, mode="onnx"
            )

            assert is_valid
            mock_load.assert_called_once_with(output_path)
            mock_check_model.assert_called_once()

    def test_validate_unknown_mode(self, sample_batch_event):
        mock_net = MockNet()
        model_wrapper = ExportModelWrapper(
            net=mock_net, dynamic_axes=EVENT_TORCH_DYNAMIC_AXES, model_type="event"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.unknown"

            with pytest.raises(ValueError, match="Unknown mode"):
                validate(model_wrapper, output_path, sample_batch_event, mode="unknown")


class TestExportAndValidate:
    """Test the complete export and validation pipeline."""

    @patch("parnassus_core.flow.export.utils.export")
    @patch("parnassus_core.flow.export.utils.validate")
    def test_export_and_validate_event(self, mock_validate, mock_export, mock_config):
        mock_model = MockFlowLightning()
        mock_dataset = MockDataset(train_type="event")

        # Mock export function
        mock_wrapper = Mock()
        mock_export.return_value = (mock_wrapper, Path("output.pt"))

        # Mock validate function to return True
        mock_validate.return_value = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.pt"

            result = export_and_validate(
                mock_model, mock_config, mock_dataset, 2, "torch", output_path
            )

            assert result is True
            mock_export.assert_called_once()
            # Validate should be called multiple times (single batch, multiple batches, different batch size)
            assert mock_validate.call_count >= 3

    @patch("parnassus_core.flow.export.utils.export")
    @patch("parnassus_core.flow.export.utils.validate")
    @pytest.mark.parametrize("impact", [True, False], ids=["impact", "no_impact"])
    def test_export_and_validate_particle(
        self, mock_validate, mock_export, mock_config_particle, impact
    ):
        mock_model = MockFlowLightning()
        mock_dataset = MockDataset(train_type="particle", impact=impact)

        # Mock export function
        mock_wrapper = Mock()
        mock_export.return_value = (mock_wrapper, Path("output.pt"))

        # Mock validate function to return True
        mock_validate.return_value = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "model.pt"

            result = export_and_validate(
                mock_model, mock_config_particle, mock_dataset, 2, "torch", output_path
            )

            assert result is True
            mock_export.assert_called_once()
            assert mock_validate.call_count >= 3
