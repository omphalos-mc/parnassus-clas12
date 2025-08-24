import numpy as np
import torch
import pytest
import tempfile
import os
import yaml

from parnassus_core.flow.utils.scaling import (
    VarTransform,
    FeatureScaler,
    Unscaler,
    get_empty_transform,
    TRANSFORM_TYPES,
    TRANSFORM_FUNCTIONS,
)

torch.manual_seed(42)

# Test data
SAMPLE_CONFIG = {
    "ptrel": {
        "type": "min_max",
        "min": 0.0,
        "max": 1.0,
        "fn": "log1p",
        "fn_scale": 1.0,
        "fn_shift": 0.0,
    },
    "eta": {
        "type": "std",
        "mean": 0.0,
        "std": 1.5,
        "fn": None,
    },
    "phi": {
        "type": "min_max_sym",
        "min": -3.14,
        "max": 3.14,
    },
    "d0": {
        "type": "std",
        "mean": 0.0,
        "std": 1.0,
        "fn": "atan",
        "fn_scale": 0.1,
        "power": 2.0,
        "clip": {"min": -10.0, "max": 10.0},
    },
}


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(SAMPLE_CONFIG, f)
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def sample_tensor_data():
    """Fixture providing sample tensor data for testing."""
    return {
        "ptrel": torch.tensor([0.1, 0.5, 0.9, 0.3]),
        "eta": torch.tensor([-2.0, 0.0, 1.5, -1.0]),
        "phi": torch.tensor([-3.0, 0.0, 3.0, 1.57]),
        "d0": torch.tensor([-5.0, 0.0, 5.0, 2.0]),
        "unknown_var": torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }


class TestVarTransform:
    """Test suite for VarTransform class."""

    @pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
    def test_init_transform_type(self, transform_type):
        """Test initialization with standard normalization type."""
        config = {
            "type": transform_type,
            "mean": 1.0,
            "std": 1.0,
            "min": 0.0,
            "max": 2.0,
        }
        transform = VarTransform("test_var", config)

        assert transform.name == "test_var"
        assert transform.type == transform_type
        match transform_type:
            case "std":
                assert transform.shift == 1.0
                assert transform.scale == 1.0
            case "min_max":
                assert transform.shift == 0.0
                assert transform.scale == 2.0
            case "min_max_sym":
                assert transform.shift == 1.0
                assert transform.scale == 1.0

    @pytest.mark.parametrize("fn", TRANSFORM_FUNCTIONS)
    def test_init_with_function(self, fn):
        """Test initialization with transformation function."""
        config = {
            "type": "std",
            "mean": 0.0,
            "std": 1.0,
            "fn": fn,
            "fn_scale": 2.0,
            "fn_shift": 1.0,
        }
        transform = VarTransform("test_var", config)

        assert transform.fn == fn
        assert transform.fn_scale == 2.0
        assert transform.fn_shift == 1.0

    def test_init_with_clip(self):
        """Test initialization with clipping."""
        config = {"type": "std", "clip": {"min": -5.0, "max": 5.0}}
        transform = VarTransform("test_var", config)

        assert transform.min_clip == -5.0
        assert transform.max_clip == 5.0

    def test_init_invalid_function(self):
        """Test initialization with invalid function raises error."""
        config = {"type": "std", "fn": "invalid_function"}

        with pytest.raises(AssertionError, match="Unknown transformation function"):
            VarTransform("test_var", config)

    def test_init_invalid_type(self):
        """Test initialization with invalid type raises error."""
        config = {"type": "invalid_type"}

        with pytest.raises(AssertionError, match="Unknown transformation type"):
            VarTransform("test_var", config)

    @pytest.mark.parametrize(
        "fn", ["log", "log1p", "sqrt", "tanh", "asinh", "atan", "pow"]
    )
    def test_transform_with_functions(self, fn):
        """Test transform method with different functions."""
        config = {"type": "std", "mean": 0.0, "std": 1.0, "fn": fn, "power": 0.5}
        transform = VarTransform("test_var", config)

        # Start with positive values for functions that need them
        if fn in ["log", "sqrt"]:
            x_original = torch.tensor([0.1, 1.0, 2.0])
        elif fn == "pow":
            x_original = torch.tensor([1.0, 2.0, -3.0])
        else:
            x_original = torch.tensor([0.5, 1.0, 1.5])

        # Transform and then inverse transform
        x_transformed = transform.transform(x_original)
        x_recovered = transform.inverse_transform(x_transformed)

        torch.testing.assert_close(x_recovered, x_original, atol=1e-5, rtol=1e-5)

    def test_transform_with_clip(self):
        """Test transform method with clipping."""
        config = {
            "type": "std",
            "mean": 0.0,
            "std": 1.0,
            "clip": {"min": -2.0, "max": 2.0},
        }
        transform = VarTransform("test_var", config)

        x = torch.tensor([-10.0, 0.0, 10.0])
        result = transform.transform(x)

        assert result[0] == -2.0
        assert result[2] == 2.0

    def test_transform_custom_shift_scale(self):
        """Test transform method with custom shift and scale."""
        config = {"type": "std", "mean": 0.0, "std": 1.0}
        transform = VarTransform("test_var", config)

        x = torch.tensor([1.0, 2.0, 3.0])
        custom_shift = 1.0
        custom_scale = 2.0

        result = transform.transform(x, shift=custom_shift, scale=custom_scale)
        expected = (x - custom_shift) / custom_scale
        x_recovered = transform.inverse_transform(
            result, shift=custom_shift, scale=custom_scale
        )

        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(x_recovered, x, atol=1e-5, rtol=1e-5)


class TestGetEmptyTransform:
    """Test suite for get_empty_transform function."""

    def test_get_empty_transform(self):
        """Test that get_empty_transform returns identity transform."""
        transform = get_empty_transform()

        assert isinstance(transform, VarTransform)
        assert transform.name == ""
        assert transform.type == "std"
        assert transform.shift == 0.0
        assert transform.scale == 1.0

    def test_empty_transform_is_identity(self):
        """Test that empty transform acts as identity."""
        transform = get_empty_transform()

        x = torch.tensor([1.0, 2.0, 3.0, -1.0])
        result = transform.transform(x)

        torch.testing.assert_close(result, x)


class TestFeatureScaler:
    """Test suite for FeatureScaler class."""

    def test_init(self, temp_yaml_file):
        """Test FeatureScaler initialization."""
        scaler = FeatureScaler(temp_yaml_file)

        assert len(scaler.scale_dict) == len(SAMPLE_CONFIG)
        assert "ptrel" in scaler.transforms
        assert "eta" in scaler.transforms

    def test_getitem(self, temp_yaml_file):
        """Test __getitem__ method."""
        scaler = FeatureScaler(temp_yaml_file)

        # Test existing feature
        ptrel_transform = scaler["ptrel"]
        assert isinstance(ptrel_transform, VarTransform)
        assert ptrel_transform.name == "ptrel"

        # Test non-existing feature (should return empty transform)
        unknown_transform = scaler["unknown_feature"]
        assert isinstance(unknown_transform, VarTransform)

    def test_transform(self, temp_yaml_file, sample_tensor_data):
        """Test transform method."""
        scaler = FeatureScaler(temp_yaml_file)

        # Create a copy to avoid modifying the original
        input_data = {k: v.clone() for k, v in sample_tensor_data.items()}
        result = scaler.transform(input_data)

        # Check that known features are transformed
        assert not torch.equal(result["ptrel"], sample_tensor_data["ptrel"])
        assert not torch.equal(result["eta"], sample_tensor_data["eta"])

        recovered = scaler.inverse_transform(result)

        for key in input_data:
            torch.testing.assert_close(
                recovered[key], sample_tensor_data[key], atol=1e-5, rtol=1e-5
            )

        # Check that unknown features remain unchanged
        torch.testing.assert_close(
            result["unknown_var"], sample_tensor_data["unknown_var"]
        )

    def test_empty_config(self):
        """Test with empty configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            temp_file = f.name

        try:
            scaler = FeatureScaler(temp_file)
            assert len(scaler.scale_dict) == 0

            # Should return empty transform for any key
            transform = scaler["any_key"]
            assert isinstance(transform, VarTransform)
        finally:
            os.unlink(temp_file)


class TestUnscaler:
    """Test suite for Unscaler class."""

    def test_init(self, temp_yaml_file):
        """Test Unscaler initialization."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        fs_vars = ["ptrel", "eta", "phi"]
        ctxt_vars = ["truth_pt", "truth_eta"]
        ctxt_global_vars = ["truth_ht", "ntruth_particles"]

        unscaler = Unscaler(
            feature_scaler=feature_scaler,
            fs_vars=fs_vars,
            ctxt_vars=ctxt_vars,
            ctxt_global_vars=ctxt_global_vars,
            sin_cos=True,
        )

        assert unscaler.feature_scaler is feature_scaler
        assert unscaler.fs_vars == fs_vars
        assert unscaler.ctxt_vars == ctxt_vars
        assert unscaler.ctxt_global_vars == ctxt_global_vars
        assert unscaler.sin_cos is True

    def test_extract_var_simple(self, temp_yaml_file):
        """Test extract_var method with simple variable."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        unscaler = Unscaler(feature_scaler, [], [], [], sin_cos=False)

        # Test data with multiple features
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        var, index_shift = unscaler.extract_var(data, "eta", var_idx=1, sin_cos=False)

        assert index_shift == 1
        assert var.shape == (2,)

    def test_extract_var_phi_with_sin_cos(self, temp_yaml_file):
        """Test extract_var method with phi variable and sin_cos=True."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        unscaler = Unscaler(feature_scaler, [], [], [], sin_cos=True)

        # Create data where indices 0,1 are sin,cos of phi
        sin_phi = torch.tensor([[0.0, 1.0], [0.707, 0.707]])  # sin values
        cos_phi = torch.tensor([[1.0, 0.0], [0.707, 0.707]])  # cos values
        data = torch.stack([sin_phi, cos_phi], dim=-1).reshape(2, 2, 2)

        var, index_shift = unscaler.extract_var(
            data, "some_phi", var_idx=0, sin_cos=True
        )

        assert index_shift == 2
        assert var.shape == (2, 2)
        # atan2(0, 1) = 0, atan2(1, 0) = π/2
        expected_first_row = torch.tensor([0.0, torch.pi / 2])
        torch.testing.assert_close(var[0], expected_first_row, atol=1e-5, rtol=1e-5)

    def test_extract_var_class(self, temp_yaml_file):
        """Test extract_var method with class variable."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        unscaler = Unscaler(feature_scaler, [], [], [], sin_cos=False)

        # Create one-hot encoded class data
        data = torch.tensor(
            [
                [0.1, 0.2, 0.8, 0.1, 0.1],  # class 2
                [0.9, 0.1, 0.1, 0.1, 0.1],  # class 0
            ]
        )

        var, index_shift = unscaler.extract_var(
            data, "particle_class", var_idx=0, sin_cos=False
        )

        assert index_shift == 5
        expected = torch.tensor([2, 0])  # argmax of each row
        torch.testing.assert_close(var, expected.long())

    @pytest.mark.parametrize("use_pf_data", [True, False])
    def test_unscale_variables(self, temp_yaml_file, use_pf_data):
        """Test unscale_variables method with basic input."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        fs_vars = ["pflow_ptrel"]
        ctxt_vars = [
            "truth_eta",
        ]  # Changed from truth_ptrel to truth_eta
        ctxt_global_vars = ["truth_ht", "pflow_ht"]

        unscaler = Unscaler(
            feature_scaler=feature_scaler,
            fs_vars=fs_vars,
            ctxt_vars=ctxt_vars,
            ctxt_global_vars=ctxt_global_vars,
            sin_cos=False,
        )

        # Random data for testing
        BS, SEQ_LEN = 2, 4
        fs_data = torch.randn(BS, SEQ_LEN, len(fs_vars))
        ctxt_data = torch.randn(BS, SEQ_LEN, len(ctxt_vars))
        ctxt_global_data = torch.randn(BS, len(ctxt_global_vars))
        if use_pf_data:
            pf_data = torch.randn(BS, SEQ_LEN, len(fs_vars))
        else:
            pf_data = None
        ctxt_dict = {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
        }

        result = unscaler.unscale_variables(fs_data, ctxt_dict, pf_data=pf_data)

        # Check that result contains expected keys
        assert "fs_pt" in result
        assert "tr_ht" in result
        assert "tr_eta" in result
        assert "pf_ht" in result

        # Check that tensors are converted to numpy
        assert isinstance(result["fs_pt"], np.ndarray)

        # Check that values are unscaled correctly (basic checks)
        # pflow_pt = np.expm1(fs_ptrel) * pflow_ht
        expected_pf_ht = ctxt_dict["ctxt_global_data"][:, 1].numpy()
        expected_fs_pt = np.expm1(fs_data.numpy())[..., 0] * expected_pf_ht.reshape(
            -1, 1
        )
        np.testing.assert_allclose(result["fs_pt"], expected_fs_pt, rtol=1e-5)
        if use_pf_data:
            expected_pf_pt = np.expm1(pf_data.numpy())[..., 0] * expected_pf_ht.reshape(
                -1, 1
            )
            np.testing.assert_allclose(result["pf_pt"], expected_pf_pt, rtol=1e-5)

    def test_unscale_variables_means_handling(self, temp_yaml_file):
        """Test handling of 'means' in ctxt_global_vars."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        ctxt_vars = ["truth_ptrel", "truth_eta", "truth_class"]
        ctxt_global_vars = ["means", "truth_ht"]

        unscaler = Unscaler(
            feature_scaler=feature_scaler,
            fs_vars=[],
            ctxt_vars=ctxt_vars,
            ctxt_global_vars=ctxt_global_vars,
            sin_cos=False,
        )

        fs_data = torch.tensor([[]])
        ctxt_dict = {
            "ctxt_global_data": torch.tensor(
                [[100.0, 200.0, 300.0]]
            ),  # means (skips 2), then truth_ht
            "ctxt_data": torch.tensor(
                [[0.5, 1.0, 0.1, 0.2, 0.8, 0.1, 0.1]]
            ),  # ptrel, eta, class (5 values)
        }

        result = unscaler.unscale_variables(fs_data, ctxt_dict)

        # Should have truth_ht but skip means processing
        assert "tr_ht" in result
        assert "means" not in result

    @pytest.mark.parametrize("sin_cos", [True, False])
    def test_unscale_variables_sin_cos_parameter(self, temp_yaml_file, sin_cos):
        """Test unscale_variables with different sin_cos settings."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        fs_vars = ["phi"]

        unscaler = Unscaler(
            feature_scaler=feature_scaler,
            fs_vars=fs_vars,
            ctxt_vars=[],
            ctxt_global_vars=[],
            sin_cos=sin_cos,
        )

        if sin_cos:
            # Need sin/cos data for phi
            fs_data = torch.tensor([[0.707, 0.707]])  # sin, cos of π/4
        else:
            fs_data = torch.tensor([[1.57]])  # Direct phi value

        result = unscaler.unscale_variables(fs_data, {})

        assert "fs_phi" in result
