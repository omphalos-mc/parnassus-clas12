import pytest
import torch
from parnassus_core.flow.modules.transformer import (
    GLU,
    SADiTLayer,
    CADiTLayer,
    DiTEncoder,
)
from parnassus_core.flow.modules.norms import LayerNorm, RMSNorm

HAS_GPU = torch.cuda.is_available()
ATTN_TYPES_GPU = {"flex", "flash", "flash-varlen"}
DEVICE = "cuda" if HAS_GPU else "cpu"

torch.manual_seed(42)


@pytest.mark.parametrize("gated", [False, True])
def test_glu_forward(gated: bool):
    x = torch.randn(2, 10, 16, device=DEVICE)
    glu = GLU(
        embed_dim=16,
        hidden_dim=32,
        activation="ReLU",
        dropout=0.1,
        bias=True,
        gated=gated,
    ).to(DEVICE)
    out = glu(x)
    assert out.shape == x.shape


MHA_CONFIG = {"attn_type": "torch"}


@pytest.mark.parametrize("norm", ["layer", "rms"])
def test_dit_norm(norm: str):
    x = torch.randn(2, 10, 16, device=DEVICE)
    ctxt = torch.randn(2, 32, device=DEVICE)
    layer = SADiTLayer(
        hidden_dim=16,
        num_heads=4,
        mlp_ratio=2.0,
        ctxt_ratio=2.0,
        norm=norm,
        mha_config=MHA_CONFIG,
    ).to(DEVICE)
    match norm:
        case "layer":
            assert isinstance(layer.norm1, LayerNorm)
        case "rms":
            assert isinstance(layer.norm1, RMSNorm)
    out = layer(x, ctxt=ctxt)
    assert out.shape == x.shape


@pytest.mark.parametrize("simplify", [False, True])
def test_sa_dit_layer(simplify: bool):
    x = torch.randn(2, 10, 16, device=DEVICE)
    ctxt = torch.randn(2, 32, device=DEVICE)
    layer = SADiTLayer(
        hidden_dim=16,
        num_heads=4,
        mlp_ratio=2.0,
        ctxt_ratio=2.0,
        simplify=simplify,
        mha_config=MHA_CONFIG,
    ).to(DEVICE)
    out = layer(x, ctxt=ctxt)
    assert out.shape == x.shape


@pytest.mark.parametrize("simplify", [False, True])
def test_ca_dit_layer(simplify: bool):
    x = torch.randn(2, 10, 16, device=DEVICE)
    ctxt = torch.randn(2, 32, device=DEVICE)
    layer = CADiTLayer(
        hidden_dim=16,
        num_heads=4,
        mlp_ratio=2.0,
        ctxt_ratio=2.0,
        simplify=simplify,
        mha_config=MHA_CONFIG,
    ).to(DEVICE)
    out = layer(x, x, ctxt=ctxt)
    assert out.shape == x.shape


@pytest.mark.parametrize("simplify", [False, True])
def test_sa_ca_equivalence_dit_layer(simplify: bool):
    x = torch.randn(2, 10, 16, device=DEVICE)
    ctxt = torch.randn(2, 32, device=DEVICE)
    sa_layer = SADiTLayer(
        hidden_dim=16,
        num_heads=4,
        mlp_ratio=2.0,
        ctxt_ratio=2.0,
        simplify=simplify,
        mha_config=MHA_CONFIG,
    ).to(DEVICE)
    ca_layer = CADiTLayer(
        hidden_dim=16,
        num_heads=4,
        mlp_ratio=2.0,
        ctxt_ratio=2.0,
        simplify=simplify,
        mha_config=MHA_CONFIG,
    ).to(DEVICE)
    for param_sa, param_ca in zip(sa_layer.parameters(), ca_layer.parameters()):
        param_ca.data = param_sa.data.clone()
    out_sa = sa_layer(x, ctxt=ctxt)
    out_ca = ca_layer(x, x, ctxt=ctxt)
    assert torch.allclose(out_sa, out_ca, atol=1e-6)


@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("do_selfattn", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("simplify", [False, True])
@pytest.mark.parametrize("mask", [False, True])
def test_dit_encoder(
    num_layers: int, do_selfattn: bool, bidirectional: bool, simplify: bool, mask: bool
):
    x_a = torch.randn(2, 10, 16, device=DEVICE).half()
    x_b = torch.randn(2, 15, 16, device=DEVICE).half()
    ctxt = torch.randn(2, 32, device=DEVICE).half()
    if mask:
        mask_a = torch.randint(0, 2, (2, 10), device=DEVICE).bool()
        mask_b = torch.randint(0, 2, (2, 15), device=DEVICE).bool()
    else:
        mask_a = None
        mask_b = None
    encoder = (
        DiTEncoder(
            hidden_dim=16,
            num_heads=4,
            num_layers=num_layers,
            mlp_ratio=2.0,
            ctxt_ratio=2.0,
            attn_type="torch",
            do_selfattn=do_selfattn,
            bidirectional=bidirectional,
            simplify=simplify,
        )
        .to(DEVICE)
        .half()
    )
    encoder.reset_parameters()
    out_a, out_b, _ = encoder(x_a, x_b, ctxt=ctxt, mask_a=mask_a, mask_b=mask_b)
    assert out_a.shape == x_a.shape
    assert out_b.shape == x_b.shape
    if not bidirectional:
        assert torch.allclose(out_b, x_b, atol=1e-6)


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
@pytest.mark.parametrize("attn_type_new", ["torch", "flash", "flex", "flash-varlen"])
def test_dit_encoder_change_backend(attn_type: str, attn_type_new: str):
    if not HAS_GPU and (attn_type in ATTN_TYPES_GPU or attn_type_new in ATTN_TYPES_GPU):
        pytest.skip(f"Skipping {attn_type} since not supported on GPU")
    x_a = torch.randn(1, 16, 128, device=DEVICE).half()
    x_b = torch.randn(1, 16, 128, device=DEVICE).half()
    ctxt = torch.randn(1, 128, device=DEVICE).half()
    encoder = (
        DiTEncoder(
            hidden_dim=128,
            num_heads=8,
            num_layers=2,
            mlp_ratio=2.0,
            ctxt_ratio=1.0,
            attn_type=attn_type,
            do_selfattn=True,
            bidirectional=True,
            simplify=False,
        )
        .to(DEVICE)
        .half()
    )
    encoder.reset_parameters()
    out_a1, out_b1, _ = encoder(x_a, x_b, ctxt=ctxt)

    encoder.set_backend(attn_type_new)
    assert encoder.attn_type == attn_type_new
    assert all(layer.attn_type == attn_type_new for layer in encoder.layers)

    out_a2, out_b2, _ = encoder(x_a, x_b, ctxt=ctxt)
    assert out_a2.shape == out_a1.shape
    assert out_b2.shape == out_b1.shape

    torch.testing.assert_close(out_a1, out_a2, atol=1e-3, rtol=1e-3)
