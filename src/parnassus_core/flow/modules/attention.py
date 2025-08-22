import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input
from torch import BoolTensor, Size, Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    _score_mod_signature,
    flex_attention,
)
from torch.nn.functional import scaled_dot_product_attention
from .norms import RMSNorm

ATTN_TYPES = {
    "torch": scaled_dot_product_attention,
    "flex": flex_attention,
    "flash": flash_attn_func,
    "flash-varlen": flash_attn_varlen_func,
}

# Which attentiom types support varlen / kv padding
VARLEN_ATTN_TYPES = [
    "torch",
    "flash-varlen",
]

# Which attention types support attention masking
ATTN_MASK_ATTN_TYPES = ["torch", "flex"]

# For now basically just defines which attention types expect (B, S, H, Dh) instead of (B, H, S, Dh)
FLASH_ATTN_TYPES = [
    "flash",
    "flash-varlen",
]


def merge_masks(
    q_mask: BoolTensor | None,
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Size,
    k_shape: Size,
    device: torch.device,
) -> BoolTensor:
    """Create a full attention mask which incoporates the padding information.
    Modified from https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/-/blob/main/salt/models/attention.py?ref_type=heads
    to use the convention that true slots are involved in computation / not masked out.
    """
    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = torch.full(q_shape[:-1], True, device=device)
        if kv_mask is None:
            kv_mask = torch.full(k_shape[:-1], True, device=device)
        merged_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

    # If attention mask exists then it must be included
    if attn_mask is not None:
        merged_mask = attn_mask & merged_mask if merged_mask is not None else attn_mask

    return merged_mask


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        value_residual: bool = False,
        qk_norm: bool = False,
        v_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "num_heads must divide dim."
        assert attn_type in ATTN_TYPES, f"Invalid attention type: {attn_type}"

        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_type = attn_type
        self.value_residual = value_residual
        self.qk_norm = qk_norm
        self.v_norm = v_norm

        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim)) if bias else None
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if self.value_residual:
            self.value_residual_mix = nn.Sequential(
                nn.Linear(dim, num_heads), nn.Sigmoid()
            )

        if self.qk_norm:
            self.q_norm = RMSNorm(dim)
            self.k_norm = RMSNorm(dim)
        if self.v_norm:
            self.v_norm = RMSNorm(dim)

        self.reset_parameters()
        self.set_backend(attn_type, torch_compile=torch_compile)

    def reset_parameters(self):
        """Initialize the parameters."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def set_backend(self, attn_type: str, torch_compile: bool = False) -> str:
        # Allow to change the attention backend after initialization, when evaluating the model

        self.attn_type = attn_type
        if attn_type not in ATTN_TYPES:
            raise ValueError(f"Invalid attention type: {attn_type}")
        self.attn = ATTN_TYPES[attn_type]

        if torch_compile or attn_type == "flex":
            self.attn = torch.compile(self.attn)
        return self.attn_type

    def separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        x = x.unflatten(-1, (num_heads, -1))  # B S D -> B S H Dh
        if self.attn_type not in FLASH_ATTN_TYPES:
            x = x.transpose(-3, -2)  # B S H Dh -> B H S Dh
        return x

    def recombine_heads(self, x: Tensor) -> Tensor:
        if self.attn_type not in FLASH_ATTN_TYPES:
            x = x.transpose(-3, -2)  # B H S Dh -> B S H Dh
        return x.flatten(-2)  # B S H Dh -> B S D

    def _prepare_qkv(
        self, q: Tensor, kv: Tensor | None = None, initial_values: dict | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Mix for value residual
        mix = None
        if self.value_residual:
            mix = self.value_residual_mix(q)
            mix = mix.unsqueeze(-1)
            if self.attn_type not in FLASH_ATTN_TYPES:
                mix = mix.transpose(-2, -3)

        # Check if the input is nested tensor
        if kv is None:
            kv = q
        q, k, v = F._in_projection_packed(
            q, kv, kv, self.in_proj_weight, self.in_proj_bias
        )  # noqa: SLF001

        # Normalize queries, keys, and values
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.v_norm:
            v = self.v_norm(v)

        # Separate into heads, output shape is (batch, head, seq, head_dim)
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # Residual connection with initial values
        if self.value_residual:
            if not initial_values:
                initial_values["v"] = v
            else:
                v = v * mix + initial_values["v"] * (1.0 - mix)
        return q, k, v

    def _flash_varlen_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
    ) -> Tensor:
        # TODO: Implement a packed version for the self attention case
        bs = q.shape[0]
        # Undo padding
        if q_mask is None:
            q_mask = torch.ones((bs, q.shape[1]), dtype=torch.bool, device=q.device)
        if kv_mask is None:
            kv_mask = torch.ones((bs, k.shape[1]), dtype=torch.bool, device=k.device)
        q_flat, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, q_mask.int())
        k_flat, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, kv_mask.int())
        v_flat, _, _, _, _ = unpad_input(v, kv_mask.int())

        out = self.attn(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )

        # Redo padding
        out = pad_input(out, indices_q, bs, max_seqlen_q)

        out = out.view(bs, -1, self.dim)

        return out

    def forward(
        self,
        q: Tensor,
        kv: Tensor | None = None,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BlockMask | BoolTensor | None = None,
        score_mod: _score_mod_signature | None = None,
        initial_values: dict | None = None,
    ) -> Tensor:
        """
        Multi-head attention forward pass.

        Parameters
        ----------
        q : Tensor
            Queries tensor of shape (B, S, D).
        kv : Tensor, optional
            Keys tensor of shape (B, S, D). If None, defaults to q.
        kv_mask : BoolTensor, optional
            Key/value mask to apply. If None, no mask is applied.
            True values indicate that a value is not padded and should partake in computation.
        attn_mask : BlockMask | BoolTensor, optional
            Attention mask to apply. If None, no mask is applied.
            True values indicate that an attention slot should partake in computation.
        score_mod : _score_mod_signature, optional
            Score modifier function for flex attention. If None, no score modifier is applied.
        initial_values : dict, optional
            Initial values for value residual connection.
        """
        if kv is None:
            # If self-attention, we use the same tensor for q, k, and v
            q_shape = kv_shape = q.shape
        else:
            # If cross-attention, we expect q and kv to be different tensors
            q_shape = q.shape
            kv_shape = kv.shape

        # Check that the specified attention backend actualy supports kv masking / jagged inputs
        if kv_mask is not None:
            msg = f"Only the backends {VARLEN_ATTN_TYPES} support kv masking"
            assert self.attn_type in VARLEN_ATTN_TYPES, msg

        if attn_mask is not None:
            msg = f"Only the backends {ATTN_MASK_ATTN_TYPES} support attention masking"
            assert self.attn_type in ATTN_MASK_ATTN_TYPES, msg

        # Prepare queries, keys, and values
        q, k, v = self._prepare_qkv(q, kv, initial_values)
        if self.attn_type == "flash-varlen":
            out = self._flash_varlen_attention(q, k, v, q_mask=q_mask, kv_mask=kv_mask)
            return self.out_proj(out)
        # Fused attention
        if self.attn_type == "flex":
            out = self.attn(q, k, v, block_mask=attn_mask, score_mod=score_mod)
        elif self.attn_type == "torch":
            attn_mask = merge_masks(
                q_mask, kv_mask, attn_mask, q_shape, kv_shape, q.device
            )
            # Have to expand the attention mask so that it is broadcasted over the head dimension
            if attn_mask is not None and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(-3)
            out = self.attn(q, k, v, attn_mask=attn_mask)

        elif self.attn_type == "flash":
            out = self.attn(q, k, v)

        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        # Get output
        out = self.recombine_heads(out)
        return self.out_proj(out)
