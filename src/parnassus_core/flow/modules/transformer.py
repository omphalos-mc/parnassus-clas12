from torch import Tensor
import torch.nn as nn

from parnassus_core.flow.modules.attention import Attention


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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CALayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mha_config: dict = None,
        mlp_ratio: float = 4.0,
        ctxt_ratio: float = 3.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm_k = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = GLU(hidden_dim, mlp_hidden_dim, bias=False, gated=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(ctxt_ratio * hidden_dim), hidden_dim * 6, bias=True),
        )
        if mha_config is None:
            mha_config = {}

        self.attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            **mha_config,
        )

    def set_backend(self, attn_type: str):
        self.attn.set_backend(attn_type)
        self.attn_type = attn_type
        return self.attn_type

    def forward(self, seq_q, seq_k, ctxt, mask_k=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(ctxt).chunk(6, dim=-1)

        seq_q = seq_q + gate_msa.unsqueeze(1) * self.attn(
            q=modulate(self.norm1(seq_q), shift_msa, scale_msa),
            kv=self.norm_k(seq_k),
            kv_mask=mask_k if self.attn_type != "flex" else None,
            attn_mask=mask_k if self.attn_type == "flex" else None,
        )

        seq_q = seq_q + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(seq_q), shift_mlp, scale_mlp)
        )
        return seq_q

    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


class CABlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mha_config=None,
        mlp_ratio=4.0,
        ctxt_ratio=3.0,
        is_last=False,
    ):
        super().__init__()
        self.is_last = is_last
        if not is_last:
            self.layer_a = CALayer(
                hidden_dim,
                num_heads,
                mha_config,
                mlp_ratio,
                ctxt_ratio,
            )
        self.layer_b = CALayer(
            hidden_dim,
            num_heads,
            mha_config,
            mlp_ratio,
            ctxt_ratio,
        )

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        if not self.is_last:
            self.layer_a.set_backend(self.attn_type)
        self.layer_b.set_backend(self.attn_type)
        return self.attn_type

    def forward(self, seq_a, seq_b, ctxt, mask_a=None, mask_b=None):
        seq_b = self.layer_b(seq_q=seq_b, seq_k=seq_a, ctxt=ctxt, mask_k=mask_a)
        if not self.is_last:
            seq_a = self.layer_a(seq_q=seq_a, seq_k=seq_b, ctxt=ctxt, mask_k=mask_b)

        return seq_a, seq_b

    def reset_parameters(self):
        if not self.is_last:
            self.layer_a.reset_parameters()
        self.layer_b.reset_parameters()


class CATransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mha_config: dict = None,
        mlp_ratio: float = 4.0,
        ctxt_ratio: float = 3.0,
        attn_type: str = "flex",
    ):
        super().__init__()
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        if mha_config is None:
            mha_config = {}
        mha_config["attn_type"] = attn_type
        self.layers = nn.ModuleList(
            [
                CABlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mha_config=mha_config,
                    mlp_ratio=mlp_ratio,
                    ctxt_ratio=ctxt_ratio,
                    is_last=(i == num_layers - 1),
                )
                for i in range(num_layers)
            ]
        )
        self.set_backend(attn_type)

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        layer: CALayer
        for layer in self.layers:
            self.attn_type = layer.set_backend(self.attn_type)
        return self.attn_type

    def forward(self, seq_a, seq_b, ctxt, mask_a=None, mask_b=None):
        for layer in self.layers:
            seq_a, seq_b = layer(seq_a, seq_b, ctxt, mask_a, mask_b)
        return seq_a, seq_b

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
