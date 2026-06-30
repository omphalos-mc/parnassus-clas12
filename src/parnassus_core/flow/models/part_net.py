from contextlib import nullcontext
from functools import partial
import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from parnassus_core.flow.modules.attention import create_padding_mask_from_kv

from parnassus_core.flow.modules import (
    NamedNetwork,
)
from parnassus_core.flow.modules import DiTEncoder


class ParticleNetwork(nn.Module):
    def __init__(
        self,
        time_embedding: nn.Module,
        fs_init_net: nn.Module,
        ctxt_networks: nn.ModuleList,
        global_embedding: nn.Module,
        encoder: DiTEncoder,
        fs_out_net: nn.Module,
        inter_out_net: nn.Module | None = None,
        ctxt_as_token: bool = False,
        num_time_tokens: int = 0,
        num_ctxt_tokens: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.time_shape = (-1, 1, 1)  # (batch_size, 1, 1)

        self.time_embedding = time_embedding
        self.fs_init_net = fs_init_net
        self.ctxt_networks = ctxt_networks
        self.global_embedding = global_embedding
        self.encoder = encoder
        self.fs_out_net = fs_out_net
        self.inter_out_net = inter_out_net

        self.ctxt_as_token = ctxt_as_token
        self.num_time_tokens = num_time_tokens
        self.num_ctxt_tokens = num_ctxt_tokens

        self.token_init_constant = 1.0
        self.hidden_dim = self.encoder.hidden_dim

        if self.ctxt_as_token:
            assert self.num_time_tokens > 0, (
                "num_time_tokens must be greater than 0 when ctxt_as_token is True"
            )
            assert self.num_ctxt_tokens > 0, (
                "num_ctxt_tokens must be greater than 0 when ctxt_as_token is True"
            )

            token_initializer = partial(
                nn.init.normal_,
                std=self.token_init_constant / math.sqrt(self.hidden_dim),
            )

            self.time_tokens = nn.Parameter(
                token_initializer(torch.empty(1, self.num_time_tokens, self.hidden_dim))
            )
            self.ctxt_tokens = nn.Parameter(
                token_initializer(torch.empty(1, self.num_ctxt_tokens, self.hidden_dim))
            )
            self.ctxt_global_tokens = nn.Parameter(
                token_initializer(torch.empty(1, self.num_ctxt_tokens, self.hidden_dim))
            )
            self.total_tokens = self.num_time_tokens + self.num_ctxt_tokens * 2

        self.context_manager = nullcontext()
        self.attn_type = self.change_backend(encoder.attn_type)

    def change_backend(self, attn_type: str):
        self.attn_type = self.encoder.set_backend(attn_type)
        if "flash" in self.attn_type:
            self.context_manager = torch.amp.autocast(  # pyright: ignore[reportPrivateImportUsage]
                enabled=True, dtype=torch.bfloat16, device_type="cuda"
            )
        else:
            self.context_manager = nullcontext()
        return self.attn_type

    def append_tokens_to_ctxt_seq(
        self,
        ctxt_seq: torch.Tensor,
        ctxt_embd_dict: dict[str, torch.Tensor],
        ctxt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_tokens = self.time_tokens + ctxt_embd_dict["time_embd"].unsqueeze(1)
        ctxt_embd = ctxt_embd_dict["ctxt_embd"]
        ctxt_denom = (
            ctxt_mask.sum(dim=1, keepdim=True).unsqueeze(-1).to(ctxt_embd.dtype)
        )
        ctxt_embd_mean = (ctxt_mask.unsqueeze(-1) * ctxt_embd).sum(
            dim=1, keepdim=True
        ) / (ctxt_denom + ctxt_embd.new_tensor(1e-6))
        ctxt_tokens = self.ctxt_tokens + ctxt_embd_mean
        ctxt_global_tokens = self.ctxt_global_tokens + ctxt_embd_dict[
            "ctxt_global_embd"
        ].unsqueeze(1)

        ctxt_seq = torch.cat(
            [time_tokens, ctxt_tokens, ctxt_global_tokens, ctxt_seq], dim=1
        ).contiguous()
        ctxt_mask = torch.cat(
            [
                torch.ones(
                    ctxt_seq.size(0),
                    self.total_tokens,
                    device=ctxt_mask.device,
                ).bool(),
                ctxt_mask,
            ],
            dim=1,
        ).contiguous()
        return ctxt_seq, ctxt_mask

    def forward(
        self,
        fs_data: torch.Tensor,
        timestep: torch.Tensor,
        ctxt_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        fs_mask = ctxt_dict["fs_mask"]
        fs_embd = self.fs_init_net(fs_data)

        ctxt_embd_dict = {}
        ctxt_net: NamedNetwork
        for ctxt_net in self.ctxt_networks:  # pyright: ignore[reportAssignmentType]
            ctxt_embd_dict[ctxt_net.input_name + "_embd"] = ctxt_net(ctxt_dict)

        ctxt_embd_dict["time_embd"] = self.time_embedding(timestep)

        global_embd = self.global_embedding(ctxt_embd_dict, ctxt_dict)
        ctxt_mask = ctxt_dict["ctxt_mask"]
        ctxt_embd = ctxt_embd_dict["ctxt_embd"]

        if self.ctxt_as_token:
            ctxt_seq, ctxt_mask = self.append_tokens_to_ctxt_seq(
                ctxt_seq=ctxt_embd, ctxt_embd_dict=ctxt_embd_dict, ctxt_mask=ctxt_mask
            )
            ctxt_embd = ctxt_seq

        fs_embd = fs_embd * fs_mask.unsqueeze(-1)
        ctxt_embd = ctxt_embd * ctxt_mask.unsqueeze(-1)

        if self.attn_type == "flex":
            B, S = fs_mask.size()
            _, S_ctxt = ctxt_mask.size()
            attn_mask_ctxt = create_block_mask(
                create_padding_mask_from_kv(ctxt_mask),
                B=B,
                H=None,
                Q_LEN=S,
                KV_LEN=S_ctxt,
                _compile=True,
            )
            attn_mask_fs = create_block_mask(
                create_padding_mask_from_kv(fs_mask),
                B=B,
                H=None,
                Q_LEN=S_ctxt,
                KV_LEN=S,
                _compile=True,
            )
        else:
            attn_mask_ctxt = ctxt_mask
            attn_mask_fs = fs_mask
        with self.context_manager:
            fs_embd, _, intermediate_output = self.encoder(
                seq_a=fs_embd,
                seq_b=ctxt_embd,
                ctxt=global_embd,
                mask_a=attn_mask_fs,
                mask_b=attn_mask_ctxt,
            )
        fs_embd = fs_embd.to(global_embd.dtype)
        if isinstance(self.fs_out_net, nn.ModuleList):
            fs_out = []
            for net in self.fs_out_net:
                fs_out.append(net(fs_embd, ctxt=global_embd))
            fs_out = torch.cat(fs_out, dim=-1)
        else:
            fs_out = self.fs_out_net(fs_embd.to(global_embd.dtype), ctxt=global_embd)
        if self.inter_out_net is not None:
            inter_out = self.inter_out_net(
                intermediate_output.to(global_embd.dtype), ctxt=global_embd
            )
            return fs_out, inter_out
        return fs_out, None
