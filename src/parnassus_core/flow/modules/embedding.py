import math
import torch as T
import torch.nn as nn


class Base2FourierFeatures(nn.Module):
    def __init__(self, start=0, stop=8, step=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step
        self.freqs_exp = T.arange(start, stop, step, dtype=T.float32)

    def forward(self, inputs):
        # Create Base 2 Fourier features
        freqs_exp = self.freqs_exp.to(inputs.device, dtype=inputs.dtype)
        freqs = 2.0**freqs_exp * 2 * T.pi
        freqs = freqs.view(
            -1, *[1] * (inputs.ndim - 1)
        )  # Reshape to match inputs shape
        # Compute features
        features = inputs.unsqueeze(1) * freqs  # Broadcasting multiplication
        features = features.transpose(1, 2).flatten(-2, -1)
        return T.cat([T.sin(features), T.cos(features)], dim=-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = T.arange(max_len).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = T.zeros(max_len, d_model)
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print(x.shape, self.pe.shape)
        x = x + self.pe[: x.size(1)].view(1, x.size(1), -1)
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_dim,
        frequency_embedding_size=256,
        max_period=10000,
        time_factor: float = 1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.time_factor = time_factor

        half = self.frequency_embedding_size // 2
        freqs = T.exp(
            -math.log(max_period) * T.arange(start=0, end=half, dtype=T.float32) / half
        )

        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t, time_factor: float = 1):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = t * time_factor

        args = t[:, None].float() * self.freqs[None]
        embedding = T.cat([T.cos(args), T.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = T.cat([embedding, T.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, time_factor=self.time_factor).type_as(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class GlobalContextEmbedding(nn.Module):
    """Constructs a global context embedding from a dictionary of tensors."""

    def __init__(self, input_names: list[str]):
        super().__init__()
        self.input_names = input_names

    def forward(
        self, ctxt_dict: dict[str, T.Tensor], mask_dict: dict[str, T.Tensor] | None
    ) -> T.Tensor:
        """
        Args:
            ctxt_dict: Dictionary of tensors containing context data.
            mask_dict: Optional dictionary of masks for the context data.
        Returns:
            A tensor representing the global context embedding.
        """
        ctxt_embd = []
        for name in self.input_names:
            if name + "_embd" in ctxt_dict:
                embd = ctxt_dict[name + "_embd"]
                if mask_dict and name in mask_dict:
                    mask = mask_dict[name + "_mask"]
                    embd = T.sum(embd * mask.unsqueeze(-1), dim=1) / T.sum(
                        mask, dim=1, keepdim=True
                    )
                elif embd.dim() == 3:
                    embd = T.mean(embd, dim=1)
                ctxt_embd.append(embd)

        return T.cat(ctxt_embd, dim=-1) if ctxt_embd else T.zeros(0)
