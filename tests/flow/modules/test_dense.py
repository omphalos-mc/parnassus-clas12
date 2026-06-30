"""
Unit tests for parnassus_core.flow.modules.dense module.:w

"""

import pytest
import torch
import numpy as np

from parnassus_core.flow.modules.dense import (
    MLPBlock,
    DenseNetwork,
    NamedNetwork,
    concat_tensors,
)


class TestMLPBlock:
    """Test suite for MLPBlock class."""

    SEQ_SHAPE = (
        5,
        32,
    )

    def test_default_parameters(self):
        """Test default parameters of MLPBlock."""
        rng = np.random.default_rng(42)
        input_dim, output_dim = rng.integers(1, 100, size=2)
        x = torch.randn(*self.SEQ_SHAPE, input_dim)
        block = MLPBlock(input_dim=input_dim, output_dim=output_dim)

        output = block(x)
        assert output.shape == (*self.SEQ_SHAPE, output_dim), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_context_is_required_when_configured(self):
        block = MLPBlock(input_dim=3, output_dim=4, ctxt_dim=2)

        with pytest.raises(ValueError, match="expecting contextual information"):
            block(torch.zeros(2, 3))

    @pytest.mark.parametrize("act", [None, "GELU"])
    def test_custom_parameters(
        self,
        act: str | None,
    ):
        rng = np.random.default_rng(42)
        input_dim = output_dim = rng.integers(1, 100)
        ctxt_dim = rng.integers(1, 20)
        ctxt = torch.randn(*self.SEQ_SHAPE, ctxt_dim)
        x = torch.randn(*self.SEQ_SHAPE, input_dim)
        block = MLPBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            ctxt_dim=ctxt_dim,
            n_layers=2,
            act=act,
            norm="LayerNorm",
            dropout=0.1,
            do_res=True,
            layer_scale=True,
            pre_norm="RMSNorm",
            init_zeros=True,
            skip_last_act=True,
        )
        output = block(x, ctxt)
        assert output.shape == (*self.SEQ_SHAPE, output_dim), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"


class TestDenseNetwork:
    """Test suite for DenseNetwork class."""

    SEQ_SHAPE = (
        5,
        32,
    )

    def test_default_parameters(self):
        """Test default parameters of MLPBlock."""
        rng = np.random.default_rng(42)
        input_dim, output_dim = rng.integers(1, 100, size=2)
        x = torch.randn(*self.SEQ_SHAPE, input_dim)
        block = DenseNetwork(input_dim=input_dim, output_dim=output_dim)

        output = block(x)
        assert output.shape == (*self.SEQ_SHAPE, output_dim), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    @pytest.mark.parametrize("act", [None, "GELU"])
    @pytest.mark.parametrize("do_out", [True, False])
    def test_custom_parameters(
        self,
        act: str | None,
        do_out: bool,
    ):
        rng = np.random.default_rng(42)
        input_dim = output_dim = rng.integers(1, 100)
        hidden_dim = int(rng.integers(1, 100))
        ctxt_dim = rng.integers(1, 20)
        ctxt = torch.randn(*self.SEQ_SHAPE, ctxt_dim)
        x = torch.randn(*self.SEQ_SHAPE, input_dim)
        block = DenseNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            ctxt_dim=ctxt_dim,
            hidden_dim=hidden_dim,
            num_blocks=2,
            n_lyr_pbk=2,
            act_h=act,
            act_o=act,
            do_out=do_out,
            norm="LayerNorm",
            dropout=0.1,
            do_res=True,
            layer_scale=True,
            pre_norm="RMSNorm",
            ctxt_in_hidden=True,
            ctxt_in_input=False,
            skip_last_act=True,
            output_init_zeros=True,
        )
        output = block(x, ctxt)
        if do_out:
            assert output.shape == (*self.SEQ_SHAPE, output_dim), (
                "Output shape mismatch"
            )
        else:
            assert output.shape == (*self.SEQ_SHAPE, hidden_dim), (
                "Output shape mismatch"
            )
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    @pytest.mark.parametrize("input_ndim, ctxt_ndim", [(3, 2), (4, 2)])
    def test_forward_context_broadcasting(self, input_ndim: int, ctxt_ndim: int):
        """Test forward pass with context broadcasting from 2D to 3D."""
        rng = np.random.default_rng(42)
        input_dims = rng.integers(1, 10, size=input_ndim).tolist()
        ctxt_dims = input_dims[:ctxt_ndim]
        network = DenseNetwork(
            input_dim=input_dims[-1],
            output_dim=10,
            ctxt_dim=ctxt_dims[-1],
        )
        input_tensor = torch.randn(input_dims)
        context_tensor = torch.randn(ctxt_dims)
        print(
            "Input shape:", input_tensor.shape, "Context shape:", context_tensor.shape
        )
        output = network(input_tensor, context_tensor)
        assert output.shape == (*input_dims[:-1], 10), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_repr(self):
        """Test string representation of DenseNetwork."""
        network = DenseNetwork(input_dim=10, hidden_dim=[32, 64], output_dim=5)
        repr_str = repr(network)
        assert "(inp):" in repr_str
        assert "(h-1):" in repr_str
        assert "(out):" in repr_str

    def test_one_line_string(self):
        """Test one-line string representation."""
        network = DenseNetwork(
            input_dim=10, hidden_dim=32, num_blocks=2, output_dim=5, ctxt_dim=3
        )
        one_line = network.one_line_string()
        assert "10(3)" in one_line  # Input with context
        assert "5" in one_line  # Output

    def test_zero_initialized_output_layer_starts_as_zero_function(self):
        network = DenseNetwork(
            input_dim=3,
            hidden_dim=5,
            output_dim=2,
            output_init_zeros=True,
        )

        output = network(torch.randn(4, 3))

        torch.testing.assert_close(output, torch.zeros(4, 2))


class TestConcatTensors:
    """Test suite for concat_tensors function."""

    def test_single_tensor(self):
        """Test concatenation with single tensor."""
        tensor = torch.randn(32, 10)
        result = concat_tensors([tensor])
        assert torch.equal(result, tensor)

    def test_mixed_2d_3d_tensors(self):
        """Test concatenation with mixed 2D and 3D tensors."""
        tensor1 = torch.randn(32, 10)  # 2D - will be unsqueezed
        tensor2 = torch.randn(32, 10, 5)  # 3D - stays as is
        tensor3 = torch.randn(32, 10)  # 2D - will be unsqueezed

        result = concat_tensors([tensor1, tensor2, tensor3])
        expected_shape = (32, 10, 7)  # 1 + 5 + 1 = 7
        assert result.shape == expected_shape

    def test_empty_list(self):
        with pytest.raises(ValueError, match="expected a non-empty list"):
            concat_tensors([])

    def test_different_batch_sizes_error(self):
        """Test that tensors with different batch sizes cause error."""
        tensor1 = torch.randn(32, 10)
        tensor2 = torch.randn(16, 10)  # Different batch size

        with pytest.raises(RuntimeError):
            concat_tensors([tensor1, tensor2])

    def test_different_sequence_lengths_error(self):
        """Test that 3D tensors with different sequence lengths cause error."""
        tensor1 = torch.randn(32, 10, 3)
        tensor2 = torch.randn(32, 5, 3)  # Different sequence length

        with pytest.raises(RuntimeError):
            concat_tensors([tensor1, tensor2])


class TestNamedNetwork:
    """Test suite for NamedNetwork class."""

    def test_forward(self):
        """Test basic forward pass."""
        dense_kwargs = {
            "input_dim": 9,
            "output_dim": 5,
            "ctxt_dim": 4,
        }  # 3 fields * 3 features each
        input_name = "particle"
        ctxt_name = "event_info"
        fields = ["x", "y", "z"]

        network = NamedNetwork(
            input_name="particle",
            dense_kwargs=dense_kwargs,
            fields=fields,
            context_name=ctxt_name,
        )

        # Create input dictionary
        inputs = {f"{input_name}_{field}": torch.randn(32, 10, 3) for field in fields}
        # Add context
        inputs[ctxt_name] = torch.randn(32, 4)

        output = network(inputs)
        assert output.shape == (32, 10, 5)
