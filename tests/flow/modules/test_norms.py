import torch

from parnassus_core.flow.modules.norms import LayerNorm, LayerScale, RMSNorm


def test_layer_scale_applies_learnable_per_feature_scale():
    layer = LayerScale(3, init_values=torch.tensor([1.0, 2.0, 3.0]))
    inputs = torch.ones(2, 3)

    output = layer(inputs)

    torch.testing.assert_close(output, torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))


def test_layer_norm_normalizes_each_feature_vector_without_affine_parameters():
    layer = LayerNorm(3)
    inputs = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

    output = layer(inputs)

    torch.testing.assert_close(output.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert list(layer.parameters()) == []


def test_rms_norm_preserves_dtype_and_normalizes_root_mean_square():
    layer = RMSNorm(4, learnable=False)
    inputs = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]],
        dtype=torch.float16,
    )

    output = layer(inputs)

    assert output.dtype == inputs.dtype
    root_mean_square = output.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(
        root_mean_square,
        torch.ones(2),
        atol=1e-3,
        rtol=1e-3,
    )
