import pickle
import numpy as np
import torch
import pytest
import tempfile
import os
import yaml
from sklearn.preprocessing import QuantileTransformer

from parnassus_core.data.utils.scaling import (
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
    def test_init_transform_type(self, transform_type, tmp_path):
        """Test initialization with standard normalization type."""
        config = {
            "type": transform_type,
            "mean": 1.0,
            "std": 1.0,
            "min": 0.0,
            "max": 2.0,
        }
        if transform_type == "quantile":
            quantile_transformer = QuantileTransformer(n_quantiles=4).fit(
                np.arange(8, dtype=np.float64).reshape(-1, 1)
            )
            qt_path = tmp_path / "quantile.pkl"
            qt_path.write_bytes(pickle.dumps(quantile_transformer))
            config["qt_path"] = str(qt_path)
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
            case "quantile":
                assert transform.qt_path == str(qt_path)
                assert isinstance(transform.qt, QuantileTransformer)

    def test_quantile_transform_round_trip(self, tmp_path):
        values = np.linspace(-3.0, 3.0, 16).reshape(-1, 1)
        quantile_transformer = QuantileTransformer(n_quantiles=8).fit(values)
        qt_path = tmp_path / "quantile.pkl"
        qt_path.write_bytes(pickle.dumps(quantile_transformer))
        transform = VarTransform(
            "test_var",
            {"type": "quantile", "qt_path": str(qt_path)},
        )
        inputs = torch.tensor([-2.5, -0.5, 1.5], dtype=torch.float64)

        recovered = transform.inverse_transform(transform.transform(inputs))

        torch.testing.assert_close(recovered, inputs)

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

    def test_class_conditioned_scaling_and_identity_fallback(self, tmp_path):
        config = {
            "class_0": {
                "eta": {"type": "std", "mean": 0.0, "std": 1.0},
            },
            "class_1": {
                "eta": {"type": "std", "mean": 10.0, "std": 2.0},
            },
        }
        config_path = tmp_path / "class_scaling.yaml"
        config_path.write_text(yaml.safe_dump(config))
        scaler = FeatureScaler(str(config_path), class_based_scaling=True)
        particle_class = torch.tensor([0, 1, 2])
        values = torch.tensor([0.0, 10.0, 5.0])

        transformed = scaler.transform(
            "eta",
            values,
            particle_class=particle_class,
        )
        recovered = scaler.inverse_transform(
            "eta",
            transformed,
            particle_class=particle_class,
        )
        fallback = scaler.transform(
            "phi",
            values,
            particle_class=particle_class,
        )

        torch.testing.assert_close(transformed, torch.tensor([0.0, 0.0, 5.0]))
        torch.testing.assert_close(recovered, values)
        torch.testing.assert_close(fallback, values)

    def test_init(self, temp_yaml_file):
        """Test Unscaler initialization."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        fs_vars = ["pflow_ptrel", "pflow_eta", "pflow_phi"]
        ctxt_vars = ["truth_pt", "truth_eta"]
        ctxt_global_vars = ["truth_ht", "pflow_ht"]

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
        assert unscaler.target_collection == "pflow"
        assert unscaler.ctxt_collection == "truth"

    def test_extract_var_simple(self, temp_yaml_file):
        """Test extract_var method with simple variable."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        unscaler = Unscaler(
            feature_scaler, ["pflow_eta"], ["truth_eta"], [], sin_cos=False
        )

        # Test data with multiple features
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        var, index_shift = unscaler.extract_var(data, "eta", var_idx=1, sin_cos=False)

        assert index_shift == 1
        assert var.shape == (2,)

    def test_extract_var_phi_with_sin_cos(self, temp_yaml_file):
        """Test extract_var method with phi variable and sin_cos=True."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        unscaler = Unscaler(
            feature_scaler, ["pflow_phi"], ["truth_eta"], [], sin_cos=True
        )

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
        unscaler = Unscaler(
            feature_scaler, ["pflow_class"], ["truth_eta"], [], sin_cos=False
        )

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

        result = unscaler.unscale_variables(
            fs_data, ctxt_dict, target_data=pf_data, unshift=False
        )

        # Check that result contains expected keys
        assert "fs_pt" in result
        assert "truth_ht" in result
        assert "truth_eta" in result
        assert "pflow_ht" in result

        # Check that tensors are converted to numpy
        assert isinstance(result["fs_pt"], np.ndarray)

        # Check that values are unscaled correctly (basic checks)
        # pflow_pt = np.expm1(fs_ptrel) * pflow_ht
        expected_pf_ht = ctxt_dict["ctxt_global_data"][:, 1].numpy()
        expected_fs_pt = np.expm1(fs_data.numpy())[..., 0] * expected_pf_ht.reshape(
            -1, 1
        )
        np.testing.assert_allclose(result["fs_pt"], expected_fs_pt, rtol=1e-5)
        if use_pf_data and pf_data is not None:
            expected_pf_pt = np.expm1(pf_data.numpy())[..., 0] * expected_pf_ht.reshape(
                -1, 1
            )
            np.testing.assert_allclose(result["pflow_pt"], expected_pf_pt, rtol=1e-5)

    def test_unscale_variables_means_handling(self, temp_yaml_file):
        """Test handling of 'means' in ctxt_global_vars."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        ctxt_vars = ["truth_ptrel", "truth_eta", "truth_class"]
        ctxt_global_vars = ["means", "truth_ht"]

        unscaler = Unscaler(
            feature_scaler=feature_scaler,
            fs_vars=["pflow_eta"],
            ctxt_vars=ctxt_vars,
            ctxt_global_vars=ctxt_global_vars,
            sin_cos=False,
        )

        fs_data = torch.zeros(1, 1, 1)
        ctxt_dict = {
            "ctxt_global_data": torch.tensor(
                [[100.0, 200.0, 300.0]]
            ),  # means (skips 2), then truth_ht
            "ctxt_data": torch.tensor(
                [[[0.5, 1.0, 0.1, 0.2, 0.8, 0.1, 0.1]]]
            ),  # ptrel, eta, class (5 values)
        }

        result = unscaler.unscale_variables(fs_data, ctxt_dict, unshift=False)

        # Should have truth_ht but skip means processing
        assert "truth_ht" in result
        assert "means" not in result

    @pytest.mark.parametrize("sin_cos", [True, False])
    def test_unscale_variables_sin_cos_parameter(self, temp_yaml_file, sin_cos):
        """Test unscale_variables with different sin_cos settings."""
        feature_scaler = FeatureScaler(temp_yaml_file)
        fs_vars = ["pflow_phi"]

        unscaler = Unscaler(
            feature_scaler=feature_scaler,
            fs_vars=fs_vars,
            ctxt_vars=["truth_eta"],
            ctxt_global_vars=[],
            sin_cos=sin_cos,
        )

        if sin_cos:
            # Need sin/cos data for phi
            fs_data = torch.tensor([[[0.707, 0.707]]])  # sin, cos of π/4
        else:
            fs_data = torch.tensor([[[1.57]]])  # Direct phi value

        result = unscaler.unscale_variables(
            fs_data,
            {
                "ctxt_data": torch.zeros(1, 1, 1),
                "ctxt_global_data": torch.zeros(1, 0),
            },
            unshift=False,
        )

        assert "fs_phi" in result

    def test_unshift_raw_vz_needs_no_pv_coords(self, temp_yaml_file):
        """Raw vz passes through the default unshift path without pv_coords."""
        unscaler = Unscaler(
            FeatureScaler(temp_yaml_file), ["pflow_vz"], ["truth_vz"], []
        )
        fs_data = torch.randn(2, 3, 1)
        ctxt_dict = {
            "ctxt_data": torch.randn(2, 3, 1),
            "ctxt_global_data": torch.zeros(2, 0),
        }

        result = unscaler.unscale_variables(fs_data, ctxt_dict)

        np.testing.assert_allclose(result["fs_vz"], fs_data[..., 0].numpy())
        np.testing.assert_allclose(
            result["truth_vz"], ctxt_dict["ctxt_data"][..., 0].numpy()
        )

    def test_unshift_vz_shifted_adds_pv_z(self, temp_yaml_file):
        """vz_shifted is unshifted to vz by adding the per-event pv_z."""
        unscaler = Unscaler(
            FeatureScaler(temp_yaml_file),
            ["pflow_vz_shifted"],
            ["truth_vz_shifted"],
            [],
        )
        fs_data = torch.randn(2, 3, 1)
        pv_coords = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, -2.0]])
        ctxt_dict = {
            "ctxt_data": torch.randn(2, 3, 1),
            "ctxt_global_data": torch.zeros(2, 0),
            "pv_coords": pv_coords,
        }

        result = unscaler.unscale_variables(fs_data, ctxt_dict)

        pv_z = pv_coords[:, 2:].numpy()
        np.testing.assert_allclose(result["fs_vz"], fs_data[..., 0].numpy() + pv_z)
        np.testing.assert_allclose(
            result["truth_vz"], ctxt_dict["ctxt_data"][..., 0].numpy() + pv_z
        )

    def test_unshift_vz_shifted_without_pv_coords_raises(self, temp_yaml_file):
        """Unshifting vz_shifted without pv_coords names the missing input."""
        unscaler = Unscaler(
            FeatureScaler(temp_yaml_file), ["pflow_vz_shifted"], ["truth_eta"], []
        )
        ctxt_dict = {
            "ctxt_data": torch.zeros(1, 1, 1),
            "ctxt_global_data": torch.zeros(1, 0),
        }

        with pytest.raises(KeyError, match="recalculate_ip=True"):
            unscaler.unscale_variables(torch.zeros(1, 1, 1), ctxt_dict)
