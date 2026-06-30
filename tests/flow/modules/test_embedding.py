"""
Unit tests for parnassus_core.flow.modules.embedding module.
"""

import pytest
import torch
import numpy as np

from parnassus_core.flow.modules.embedding import (
    FourierBase2Embedding,
    PositionalEmbedding,
    TimestepEmbedder,
    GlobalContextEmbedding,
)


class TestFourierBase2Embedding:
    """Test suite for FourierBase2Embedding class."""

    SEQ_SHAPE = (5, 32)

    def test_default_parameters(self):
        """Test default parameters of FourierBase2Embedding."""
        rng = np.random.default_rng(42)
        input_dim = rng.integers(1, 10)
        x = torch.randn(*self.SEQ_SHAPE, input_dim)
        embedding = FourierBase2Embedding()

        output = embedding(x)
        expected_features = len(torch.arange(0, 8, 1)) * 2 * input_dim  # sin + cos
        assert output.shape == (*self.SEQ_SHAPE, expected_features), (
            "Output shape mismatch"
        )
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    @pytest.mark.parametrize("input_dim", [1, 5])
    @pytest.mark.parametrize("start", [0, 2])
    @pytest.mark.parametrize("stop", [4, 12])
    @pytest.mark.parametrize("step", [1, 2])
    def test_custom_parameters(self, input_dim: int, start: int, stop: int, step: int):
        """Test custom parameters of FourierBase2Embedding."""
        if start >= stop:
            pytest.skip("Invalid parameter combination: start >= stop")

        x = torch.randn(*self.SEQ_SHAPE, input_dim)
        embedding = FourierBase2Embedding(start=start, stop=stop, step=step)

        output = embedding(x)
        expected_freqs = len(torch.arange(start, stop, step))
        expected_features = expected_freqs * 2 * input_dim  # sin + cos
        assert output.shape == (*self.SEQ_SHAPE, expected_features), (
            "Output shape mismatch"
        )
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_repr(self):
        """Test string representation."""
        embedding = FourierBase2Embedding(start=1, stop=6, step=2)
        repr_str = repr(embedding)
        assert "FourierBase2Embedding" in repr_str
        assert "start=1" in repr_str
        assert "stop=6" in repr_str
        assert "step=2" in repr_str


class TestPositionalEmbedding:
    """Test suite for PositionalEmbedding class."""

    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    @pytest.mark.parametrize("seq_len", [1, 5])
    def test_forward(self, seq_len: int, dropout: float):
        """Test default parameters of PositionalEmbedding."""
        d_model = 64
        batch_size = 8
        x = torch.randn(batch_size, seq_len, d_model)
        embedding = PositionalEmbedding(d_model, dropout=dropout)

        output = embedding(x)
        assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_max_len_constraint(self):
        """Test that sequences longer than max_len work correctly."""
        d_model = 32
        max_len = 100
        seq_len = 50  # Shorter than max_len to test normal operation
        batch_size = 2

        x = torch.randn(batch_size, seq_len, d_model)
        embedding = PositionalEmbedding(d_model, max_len=max_len)

        # Should work fine for sequences shorter than max_len
        output = embedding(x)
        assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"

        # Test that we can't exceed max_len (this should raise an error)
        long_seq_len = max_len + 50
        x_long = torch.randn(batch_size, long_seq_len, d_model)

        with pytest.raises(RuntimeError):
            embedding(x_long)

    def test_repr(self):
        """Test string representation."""
        embedding = PositionalEmbedding(d_model=128, dropout=0.2, max_len=2000)
        repr_str = repr(embedding)
        assert "PositionalEmbedding" in repr_str
        assert "d_model=128" in repr_str
        assert "dropout=0.2" in repr_str
        assert "max_len=2000" in repr_str

    def test_zero_input_returns_registered_positional_encoding(self):
        embedding = PositionalEmbedding(d_model=4, dropout=0.0, max_len=3)
        inputs = torch.zeros(2, 3, 4)

        output = embedding(inputs)

        torch.testing.assert_close(
            output,
            embedding.pe.unsqueeze(0).expand(2, -1, -1),
        )


class TestTimestepEmbedder:
    """Test suite for TimestepEmbedder class."""

    @pytest.mark.parametrize("hidden_dim", [8, 32])
    @pytest.mark.parametrize("frequency_embedding_size", [16, 21])
    def test_forward(self, hidden_dim: int, frequency_embedding_size: int):
        """Test custom parameters of TimestepEmbedder."""
        batch_size = 8
        t = torch.randn(batch_size)
        embedder = TimestepEmbedder(
            hidden_dim=hidden_dim, frequency_embedding_size=frequency_embedding_size
        )

        output = embedder(t)
        assert output.shape == (batch_size, hidden_dim), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    @pytest.mark.parametrize("max_period", [1000, 100000])
    @pytest.mark.parametrize("time_factor", [0.5, 2.0])
    def test_time_parameters(self, max_period: int, time_factor: float):
        """Test time-related parameters."""
        hidden_dim = 64
        batch_size = 4
        t = torch.randn(batch_size)
        embedder = TimestepEmbedder(
            hidden_dim=hidden_dim, max_period=max_period, time_factor=time_factor
        )

        output = embedder(t)
        assert output.shape == (batch_size, hidden_dim), "Output shape mismatch"

    def test_repr(self):
        """Test string representation."""
        embedder = TimestepEmbedder(
            hidden_dim=256,
            frequency_embedding_size=512,
            max_period=50000,
            time_factor=0.5,
        )
        repr_str = repr(embedder)
        assert "TimestepEmbedder" in repr_str
        assert "hidden_dim=256" in repr_str
        assert "frequency_embedding_size=512" in repr_str
        assert "max_period=50000" in repr_str
        assert "time_factor=0.5" in repr_str


class TestGlobalContextEmbedding:
    """Test suite for GlobalContextEmbedding class."""

    @pytest.mark.parametrize("use_mask", [True, False])
    def test_forward(self, use_mask: bool):
        """Test forward pass with embedding data."""
        input_names = ["jet", "track"]
        embedding = GlobalContextEmbedding(input_names)

        batch_size = 8
        seq_len = 20
        embd_dim = 32

        ctxt_dict = {
            "jet_embd": torch.randn(batch_size, embd_dim),
            "track_embd": torch.randn(batch_size, seq_len, embd_dim * 2),
        }
        mask_dict = None
        if use_mask:
            mask_dict = {
                "track_mask": torch.randint(0, 2, (batch_size, seq_len)),
            }

        output = embedding(ctxt_dict, mask_dict)
        expected_dim = 3 * embd_dim  # Concatenated embeddings
        assert output.shape == (batch_size, expected_dim), "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_zero_mask(self):
        """Test forward pass with embedding data."""
        input_names = ["track"]
        embedding = GlobalContextEmbedding(input_names)

        batch_size = 8
        seq_len = 20
        embd_dim = 32

        ctxt_dict = {
            "track_embd": torch.randn(batch_size, seq_len, embd_dim * 2),
        }
        mask_dict = {
            "track_mask": torch.full((batch_size, seq_len), 0),
        }

        output = embedding(ctxt_dict, mask_dict)
        expected_dim = 2 * embd_dim  # Concatenated embeddings
        assert output.shape == (batch_size, expected_dim), "Output shape mismatch"
        assert torch.all(output == 0), "Output should be zero when mask is all zeros"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_masked_sequence_embedding_is_mean_over_valid_entries(self):
        embedding = GlobalContextEmbedding(["track"])
        track = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])

        output = embedding(
            {"track_embd": track},
            {"track_mask": mask},
        )

        torch.testing.assert_close(output, torch.tensor([[3.0, 5.0]]))

    def test_repr(self):
        """Test string representation."""
        input_names = ["feature1", "feature2", "feature3"]
        embedding = GlobalContextEmbedding(input_names)
        repr_str = repr(embedding)
        assert "GlobalContextEmbedding" in repr_str
        assert "feature1" in repr_str
        assert "feature2" in repr_str
        assert "feature3" in repr_str
