from typing import Any

import torch.nn as nn
from torch import Tensor

from parnassus_core.flow.modules import DiTEncoder, NamedNetwork


class SANamedNetwork(NamedNetwork):
    """NamedNetwork followed by self-attention across the embedded set.

    Mixes information between set elements (e.g. truth particles) before any
    downstream pooling. Padded elements are excluded via ``<input_name>_mask``.

    Parameters
    ----------
    input_name : str
        Name of the feature / object that will be embedded, e.g. ``ctxt``.
    dense_kwargs : dict[str, Any]
        Arguments of the per-element DenseNetwork. Its ``output_dim`` must equal
        ``sa_encoder.hidden_dim``.
    fields : list[str]
        Fields of ``input_name`` concatenated to form the feature vector.
    sa_encoder : DiTEncoder
        Unconditioned (``no_modulation=True``), unidirectional encoder applied as
        self-attention over the embedded elements.
    output_dim : int | None, optional
        If set and different from ``sa_encoder.hidden_dim``, a linear layer
        projects the attended elements to this width.
    context_name : str | None, optional
        Optional context input of the DenseNetwork.
    """

    def __init__(
        self,
        input_name: str,
        dense_kwargs: dict[str, Any],
        fields: list[str],
        sa_encoder: DiTEncoder,
        output_dim: int | None = None,
        context_name: str | None = None,
    ):
        super().__init__(input_name, dense_kwargs, fields, context_name)
        assert sa_encoder.no_modulation and not sa_encoder.bidirectional, (
            "sa_encoder must be unconditioned (no_modulation=True) and unidirectional"
        )
        self.sa_encoder = sa_encoder
        self.out_proj = (
            nn.Linear(sa_encoder.hidden_dim, output_dim)
            if output_dim and output_dim != sa_encoder.hidden_dim
            else nn.Identity()
        )

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        x = super().forward(inputs)
        mask = inputs[f"{self.input_name}_mask"]
        x, _, _ = self.sa_encoder(seq_a=x, seq_b=x, ctxt=None, mask_a=mask, mask_b=mask)
        return self.out_proj(x)
