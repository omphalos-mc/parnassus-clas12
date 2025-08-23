# Based on source code from: https://github.com/samvanstroud/hepattn
import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from parnassus_core.flow.modules.attention import (
    ATTN_MASK_ATTN_TYPES,
    VARLEN_ATTN_TYPES,
    Attention,
)

HAS_GPU = torch.cuda.is_available()
ATTN_TYPES_GPU = {"flex", "flash", "flash-varlen"}
DEVICE = "cuda" if HAS_GPU else "cpu"

torch.manual_seed(42)


def create_flex_attn_mask(attn_mask):
    def padding(b, h, q_idx, kv_idx):
        return attn_mask[b, q_idx, kv_idx]

    return padding


def copy_attention_weights(src: Attention, dst: Attention):
    dst.in_proj_weight.data.copy_(src.in_proj_weight.data)
    if src.in_proj_bias is not None:
        dst.in_proj_bias.data.copy_(src.in_proj_bias.data)
    dst.out_proj.weight.data.copy_(src.out_proj.weight.data)
    if src.out_proj.bias is not None:
        dst.out_proj.bias.data.copy_(src.out_proj.bias.data)


# Choose primes so we don't get accidental broadcasting
# NOTE: We need enough keys/queries such that it is improbable that an entire k/q slot is masked
# if that happens then nans are produced
@pytest.mark.parametrize("batch_size", [1, 9])
@pytest.mark.parametrize("q_len", [128])
@pytest.mark.parametrize("kv_len", [None, 128, 150])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("kv_masking", [False, True])
@pytest.mark.parametrize("attn_masking", [False, True])
@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
def test_attention_consistency(
    batch_size, dim, num_heads, bias, q_len, kv_len, attn_masking, kv_masking, attn_type
):
    if not HAS_GPU and attn_type in ATTN_TYPES_GPU:
        pytest.skip(f"Skipping {attn_type} test as GPU is not available")
    if attn_type == "flash-varlen" and kv_len is not None:
        pytest.skip("flash-varlen does not support cross attention")

    # Generate random input tensors
    q = torch.randn(batch_size, q_len, dim, dtype=torch.float16, device=DEVICE)
    self_attn = False
    # If kv_len is none, test self attenttion
    if kv_len:
        kv = torch.randn(batch_size, kv_len, dim, dtype=torch.float16, device=DEVICE)
    else:
        kv = q
        self_attn = True

    kv_mask = None
    if not self_attn and kv_masking and attn_type in VARLEN_ATTN_TYPES:
        kv_mask = (
            torch.randn(batch_size, kv.shape[-2], dtype=torch.float16, device=DEVICE)
            >= 0.0
        )

    attn_mask = None
    if attn_masking and attn_type in ATTN_MASK_ATTN_TYPES:
        attn_mask = (
            torch.randn(
                batch_size,
                q.shape[-2],
                kv.shape[-2],
                dtype=torch.float16,
                device=DEVICE,
            )
            >= 0.0
        )

    # Initialize attention layers
    attention_layer = (
        Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type=attn_type)
        .to(DEVICE)
        .half()
    )
    mha_layer = (
        nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True
        )
        .to(DEVICE)
        .half()
    )

    # Synchronize weights for comparison
    copy_attention_weights(attention_layer, mha_layer)

    # Torch MHA expects (batch * heads, num_q, num_k) for attention mask, so have to repeat
    # It also expects the masking convention to be backwards
    torch_attn_mask = (
        ~attn_mask.repeat_interleave(num_heads, dim=0)
        if attn_mask is not None
        else None
    )
    torch_kv_mask = ~kv_mask if kv_mask is not None else None
    mha_out, _ = mha_layer(
        q, kv, kv, key_padding_mask=torch_kv_mask, attn_mask=torch_attn_mask
    )

    if attn_type == "flash-varlen":
        if kv_mask is None:
            kv_mask = torch.full(
                (batch_size, kv.shape[-2]), True, dtype=torch.bool, device=DEVICE
            )
    if attn_type == "flex" and attn_mask is not None:
        B, Q_LEN = q.shape[0], q.shape[1]
        KV_LEN = kv.shape[1]
        attn_mask = create_block_mask(
            create_flex_attn_mask(attn_mask),
            B=B,
            H=None,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=DEVICE,
        )

    # Compute outputs
    custom_out = attention_layer(
        q,
        kv if not self_attn else None,
        kv_mask=kv_mask,
        attn_mask=attn_mask,
    )
    # Compare outputs
    torch.testing.assert_close(custom_out, mha_out, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_flex_dynamic():
    # generate inputs
    xs = [
        torch.randn(1, i, 128, dtype=torch.float16, device=DEVICE)
        for i in range(100, 110)
    ]

    # Initialize attention layers
    attn = (
        Attention(
            dim=128, num_heads=8, attn_type="flex", torch_compile=True, bias=False
        )
        .to(DEVICE)
        .half()
    )

    # loop over inputs
    for x in xs:
        out = attn(x, x)
        assert out.shape == x.shape


@pytest.mark.gpu
def test_qkv_norm():
    # generate inputs
    x = torch.randn(2, 64, 128, dtype=torch.float16, device=DEVICE)

    # Initialize attention layers
    attn = (
        Attention(
            dim=128,
            num_heads=8,
            attn_type="flex",
            torch_compile=True,
            bias=False,
            qk_norm=True,
            v_norm=True,
        )
        .to(DEVICE)
        .half()
    )

    out = attn(x, x)
    assert out.shape == x.shape


@pytest.mark.gpu
def test_fuse_mlp():
    # generate inputs
    x = torch.randn(2, 16, 64, dtype=torch.float16, device=DEVICE)

    # Initialize attention layers
    attn = (
        Attention(
            dim=64,
            num_heads=4,
            attn_type="torch",
            bias=False,
            fuse_mlp=True,
            mlp_activation="SiLU",
            mlp_ratio=2.0,
        )
        .to(DEVICE)
        .half()
    )

    out = attn(x)

    assert out.shape == x.shape


@pytest.mark.gpu
def test_value_residual():
    # generate inputs
    x = torch.randn(2, 64, 128, dtype=torch.float16, device=DEVICE)
    initial_values = {}
    # Initialize attention layers
    attn = (
        Attention(
            dim=128,
            num_heads=8,
            attn_type="flex",
            torch_compile=True,
            bias=False,
            value_residual=True,
        )
        .to(DEVICE)
        .half()
    )

    out = attn(x, x, initial_values=initial_values)
    out = attn(out, out, initial_values=initial_values)
    assert out.shape == x.shape


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
def test_self_attention(batch_size, seq_len, dim, num_heads, bias, attn_type):
    if not HAS_GPU and attn_type in ATTN_TYPES_GPU:
        pytest.skip(f"Skipping {attn_type} test as GPU is not available")

    # Generate random input tensor
    qkv = torch.randn(batch_size, seq_len, dim, dtype=torch.float16, device=DEVICE)

    # Initialize attention layers
    attn_torch = (
        Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type="torch")
        .to(DEVICE)
        .half()
    )
    attn_test = (
        Attention(dim=dim, num_heads=num_heads, bias=bias, attn_type=attn_type)
        .to(DEVICE)
        .half()
    )

    # Synchronize weights for comparison
    copy_attention_weights(attn_torch, attn_test)

    # Compute outputs
    out_torch = attn_torch(qkv)
    if attn_type == "flash-varlen":
        kv_mask = torch.full(
            (batch_size, seq_len), True, dtype=torch.bool, device=DEVICE
        )
    else:
        kv_mask = None

    out_test = attn_test(qkv, kv_mask=kv_mask)
    out_test_sa = attn_test(qkv, qkv, kv_mask=kv_mask)

    # Check that outputs are consistent
    torch.testing.assert_close(out_torch, out_test, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_test, out_test_sa, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex"])
def test_cross_attention(attn_type):
    if not HAS_GPU and attn_type in ATTN_TYPES_GPU:
        pytest.skip(f"Skipping {attn_type} test as GPU is not available")

    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device=DEVICE)
    kv = torch.randn(1, 256, 128, dtype=torch.float16, device=DEVICE)

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type=attn_type).to(DEVICE).half()

    # Compute outputs
    out = attn(q, kv)

    # Check output shape
    assert out.shape == q.shape


@pytest.mark.parametrize("attn_type", ["torch", "flash", "flex", "flash-varlen"])
@pytest.mark.parametrize("attn_type_new", ["torch", "flash", "flex", "flash-varlen"])
def test_attention_change_backend(attn_type, attn_type_new):
    if not HAS_GPU and (attn_type in ATTN_TYPES_GPU or attn_type_new in ATTN_TYPES_GPU):
        pytest.skip("Skipping GPU-specific test on CPU-only environment")
    # Generate random input tensors
    q = torch.randn(1, 128, 128, dtype=torch.float16, device=DEVICE)
    kv = torch.randn(1, 128, 128, dtype=torch.float16, device=DEVICE)

    q_in = q_out = q
    kv_in = kv_out = kv

    # Initialize attention layers
    attn = Attention(dim=128, num_heads=8, attn_type=attn_type).to(DEVICE).half()

    if attn_type in VARLEN_ATTN_TYPES and attn_type_new in VARLEN_ATTN_TYPES:
        kv_mask = torch.full((1, kv.shape[-2]), True, dtype=torch.bool, device=DEVICE)
    else:
        kv_mask = None

    # Compute outputs
    out = attn(q_in, kv_in, kv_mask=kv_mask)

    # Change backend
    attn.set_backend(attn_type_new)
    assert attn.attn_type == attn_type_new, (
        f"Expected {attn_type_new}, got {attn.attn_type}"
    )

    if attn_type_new == "flash-varlen":
        kv_mask = torch.full((1, kv.shape[-2]), True, dtype=torch.bool, device=DEVICE)
    out_new = attn(q_out, kv_out, kv_mask=kv_mask)

    # Check that the outputs are consistent
    torch.testing.assert_close(out, out_new, atol=1e-3, rtol=1e-3)
