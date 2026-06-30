from dataclasses import dataclass
from functools import partial
from typing import Literal, cast
from torch import Tensor
import torch.nn as nn

from torch.nn.attention.flex_attention import BlockMask

from parnassus_core.flow.modules.attention import Attention
from parnassus_core.flow.modules.norms import RMSNorm, LayerNorm


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        """Dense update with gated linear unit.

        See [2002.05202](https://arxiv.org/abs/2002.05202).

        Parameters
        ----------
        embed_dim : int
            Dimension of the input and output.
        hidden_dim : int | None, optional
            Dimension of the hidden layer. If None, defaults to embed_dim * 2.
        activation : str, optional
            Activation function.
        dropout : float, optional
            Dropout rate.
        bias : bool, optional
            Whether to include bias in the linear layers.
        gated : bool, optional
            Whether to gate the output of the hidden layer.
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

    def modulate(self, x: Tensor) -> Tensor:
        return x * (1 + self.scale) + self.shift


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, ctxt_ratio: float = 3.0):
        super().__init__()
        self.dim = dim
        self.ctxt_ratio = ctxt_ratio
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(
            int(dim * ctxt_ratio), int(self.multiplier * dim), bias=True
        )

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

    def reset_parameters(self):
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)


class DiTLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mha_config: dict | None = None,
        mlp_ratio: float = 4.0,
        ctxt_ratio: float = 3.0,
        simplify: bool = False,
        norm: Literal["rms", "layer", "rms-free"] = "rms",
        no_modulation: bool = False,
    ):
        super().__init__()
        self.simplify = simplify
        self.no_modulation = no_modulation
        if norm == "layer":
            norm_layer = LayerNorm
            self.norm_k = LayerNorm(hidden_dim)
        elif norm == "rms":
            norm_layer = RMSNorm
            self.norm_k = RMSNorm(hidden_dim)
        elif norm == "rms-free":
            norm_layer = partial(RMSNorm, learnable=False)
            self.norm_k = RMSNorm(hidden_dim)
        else:
            raise ValueError(
                f"Unknown norm type: {norm}, only 'rms', 'layer', and 'rms-free' are supported."
            )
        self.norm1 = norm_layer(hidden_dim)
        if not simplify:
            # self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = GLU(hidden_dim, mlp_hidden_dim, bias=False, gated=True)
        if not no_modulation:
            self.adaLN_modulation = Modulation(
                hidden_dim, double=not simplify, ctxt_ratio=ctxt_ratio
            )
        if mha_config is None:
            mha_config = {}

        self.attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            fuse_mlp=simplify,
            mlp_ratio=mlp_ratio,
            **mha_config,
        )
        self.set_backend(mha_config.get("attn_type", "torch"))

    def set_backend(self, attn_type: str):
        self.attn.set_backend(attn_type)
        self.attn_type = attn_type
        return self.attn_type

    def forward_no_modulation(
        self,
        seq_q: Tensor,
        seq_k: Tensor,
        mask_k: Tensor | BlockMask | None = None,
    ):
        seq_q = seq_q + self.attn(
            q=self.norm1(seq_q),
            kv=self.norm_k(seq_k),
            kv_mask=mask_k if self.attn_type != "flex" else None,
            attn_mask=mask_k if self.attn_type == "flex" else None,
        )
        if not self.simplify:
            seq_q = seq_q + self.mlp(self.norm2(seq_q))
        return seq_q

    def forward(
        self,
        seq_q: Tensor,
        seq_k: Tensor,
        ctxt: Tensor,
        mask_k: Tensor | BlockMask | None = None,
    ):
        if self.no_modulation:
            return self.forward_no_modulation(seq_q, seq_k, mask_k)

        # Common stuff
        mod_mha, mod_mlp = self.adaLN_modulation(ctxt)
        seq_q_mod = mod_mha.modulate(self.norm1(seq_q))
        seq_k = self.norm_k(seq_k)

        if self.simplify:
            return seq_q + mod_mha.gate * self.attn(
                q=seq_q_mod,
                kv=seq_k,
                kv_mask=mask_k if self.attn_type != "flex" else None,
                attn_mask=mask_k if self.attn_type == "flex" else None,
            )

        seq_q = seq_q + mod_mha.gate * self.attn(
            q=seq_q_mod,
            kv=seq_k,
            kv_mask=mask_k if self.attn_type != "flex" else None,
            attn_mask=mask_k if self.attn_type == "flex" else None,
        )
        seq_q = seq_q + mod_mlp.gate * self.mlp(mod_mlp.modulate(self.norm2(seq_q)))
        return seq_q

    def reset_parameters(self):
        if not self.no_modulation:
            self.adaLN_modulation.reset_parameters()


class SADiTLayer(DiTLayer):
    def forward(
        self, seq: Tensor, ctxt: Tensor, mask: Tensor | BlockMask | None = None
    ):
        return super().forward(seq, seq, ctxt, mask)


class CADiTLayer(DiTLayer):
    def forward(
        self,
        seq_q: Tensor,
        seq_k: Tensor,
        ctxt: Tensor,
        mask_k: Tensor | BlockMask | None = None,
    ):
        return super().forward(seq_q, seq_k, ctxt, mask_k)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mha_config=None,
        mlp_ratio=4.0,
        ctxt_ratio=3.0,
        bidirectional=False,
        do_selfattn=False,
        simplify=False,
        norm: Literal["rms", "layer", "rms-free"] = "rms",
        no_modulation=False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.do_selfattn = do_selfattn
        self.no_modulation = no_modulation

        kwargs = {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "mha_config": mha_config,
            "mlp_ratio": mlp_ratio,
            "ctxt_ratio": ctxt_ratio,
            "simplify": simplify,
            "norm": norm,
            "no_modulation": no_modulation,
        }

        if do_selfattn:
            self.sa_layer_a = SADiTLayer(
                **kwargs,
            )
            if bidirectional:
                self.sa_layer_b = SADiTLayer(
                    **kwargs,
                )
        self.ca_layer_a = CADiTLayer(
            **kwargs,
        )
        if bidirectional:
            self.ca_layer_b = CADiTLayer(
                **kwargs,
            )

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        if self.do_selfattn:
            self.sa_layer_a.set_backend(attn_type)
            if self.bidirectional:
                self.sa_layer_b.set_backend(attn_type)
        self.ca_layer_a.set_backend(attn_type)
        if self.bidirectional:
            self.ca_layer_b.set_backend(attn_type)
        return self.attn_type

    def forward(
        self,
        seq_a: Tensor,
        seq_b: Tensor,
        ctxt: Tensor,
        mask_a: Tensor | BlockMask | None = None,
        mask_b: Tensor | BlockMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the DiT block. It assumes that the seq_a is updated first and then seq_b is updated, if bidirectional is True.

        Parameters
        ----------
        seq_a : torch.Tensor
            The first input sequence.
        seq_b : torch.Tensor
            The second input sequence.
        ctxt : torch.Tensor
            The context tensor used for modulation.
        mask_a : torch.Tensor | BlockMask | None, optional
            The attention mask for seq_a.
        mask_b : torch.Tensor | BlockMask | None, optional
            The attention mask for seq_b.
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The updated seq_a and seq_b tensors.

        """
        if self.do_selfattn:
            seq_a = self.sa_layer_a(seq=seq_a, ctxt=ctxt, mask=mask_a)
        seq_a = self.ca_layer_a(seq_q=seq_a, seq_k=seq_b, ctxt=ctxt, mask_k=mask_b)
        if self.bidirectional:
            if self.do_selfattn:
                seq_b = self.sa_layer_b(seq=seq_b, ctxt=ctxt, mask=mask_b)
            seq_b = self.ca_layer_b(seq_q=seq_b, seq_k=seq_a, ctxt=ctxt, mask_k=mask_a)
        return seq_a, seq_b

    def reset_parameters(self):
        if self.do_selfattn:
            self.sa_layer_a.reset_parameters()
            if self.bidirectional:
                self.sa_layer_b.reset_parameters()
        self.ca_layer_a.reset_parameters()
        if self.bidirectional:
            self.ca_layer_b.reset_parameters()


class DiTEncoder(nn.Module):
    """
    DiT Encoder consisting of multiple DiT blocks.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layers.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of DiT blocks.
    mha_config : dict, optional
        Configuration for the multi-head attention.
    mlp_ratio : float, optional
        Ratio of the MLP hidden dimension to the hidden dimension.
    ctxt_ratio : float, optional
        Ratio of the context dimension to the hidden dimension.
    attn_type : str, optional
        Type of attention mechanism to use ('torch', 'flex', etc.).
    do_selfattn : bool, optional
        Whether to include self-attention layers.
    bidirectional : bool, optional
        Whether the cross-attention is bidirectional.
    simplify : bool, optional
        Whether to use the simplified architecture.
    norm : Literal['rms', 'layer'], optional
        Type of normalization to use ('rms' or 'layer').
    itermediate_idx_to_return : int, optional
        Index of the intermediate layer to return outputs from. If -1, no intermediate output is returned.

    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mha_config: dict | None = None,
        mlp_ratio: float = 4.0,
        ctxt_ratio: float = 3.0,
        attn_type: str = "torch",
        do_selfattn: bool = False,
        bidirectional: bool = True,
        simplify: bool = False,
        norm: Literal["rms", "layer", "rms-free"] = "rms",
        intermediate_idx_to_return: int = -1,
        no_modulation: bool = False,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.do_selfattn = do_selfattn
        self.bidirectional = bidirectional
        self.simplify = simplify
        self.no_modulation = no_modulation

        # For returning intermediate layer outputs
        self.return_intermediate = intermediate_idx_to_return >= 0
        self.intermediate_idx = intermediate_idx_to_return

        if mha_config is None:
            mha_config = {}
        mha_config["attn_type"] = attn_type
        self.layers = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mha_config=mha_config,
                    mlp_ratio=mlp_ratio,
                    ctxt_ratio=ctxt_ratio,
                    bidirectional=bidirectional and i != num_layers - 1,
                    do_selfattn=do_selfattn,
                    simplify=simplify,
                    norm=norm,
                    no_modulation=no_modulation,
                )
                for i in range(num_layers)
            ]
        )
        self.set_backend(attn_type)
        # RESET PARAMETERS
        self.reset_parameters()

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        for layer in self.layers:
            layer = cast(DiTBlock, layer)
            self.attn_type = layer.set_backend(self.attn_type)
        return self.attn_type

    def forward(
        self,
        seq_a: Tensor,
        seq_b: Tensor,
        ctxt: Tensor,
        mask_a: Tensor | BlockMask | None = None,
        mask_b: Tensor | BlockMask | None = None,
    ):
        intermediate_output = None
        for i, layer in enumerate(self.layers):
            seq_a, seq_b = layer(seq_a, seq_b, ctxt, mask_a, mask_b)
            if self.return_intermediate and i == self.intermediate_idx:
                intermediate_output = seq_a
        return seq_a, seq_b, intermediate_output

    def reset_parameters(self):
        for layer in self.layers:
            layer = cast(DiTBlock, layer)
            layer.reset_parameters()
