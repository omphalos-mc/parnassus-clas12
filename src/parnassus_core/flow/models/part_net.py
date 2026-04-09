from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from parnassus_core.flow.modules import (
    NamedNetwork,
)


class ParticleNetwork(nn.Module):
    def __init__(
        self,
        time_embedding: nn.Module,
        fs_init_net: nn.Module,
        ctxt_networks: nn.ModuleList,
        global_embedding: nn.Module,
        encoder: nn.Module,
        fs_out_net: nn.Module,
        inter_out_net: nn.Module | None = None,
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

        self.context_manager = nullcontext()
        self.attn_type = self.change_backend(encoder.attn_type)

    def create_padding_mask(self, pads_kv):
        S = pads_kv.size(1)

        def padding(b, h, q_idx, kv_idx):
            is_valid_idx = kv_idx < S
            safe_kv_idx = torch.clamp(kv_idx, max=S - 1)
            return is_valid_idx & pads_kv[b, safe_kv_idx]

        return padding

    def change_backend(self, attn_type: str):
        self.attn_type = self.encoder.set_backend(attn_type)
        if "flash" in self.attn_type:
            self.context_manager = torch.amp.autocast(
                enabled=True, dtype=torch.bfloat16, device_type="cuda"
            )
        else:
            self.context_manager = nullcontext()
        return self.attn_type

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
        for ctxt_net in self.ctxt_networks:
            ctxt_embd_dict[ctxt_net.input_name + "_embd"] = ctxt_net(ctxt_dict)

        ctxt_embd_dict["time_embd"] = self.time_embedding(timestep)

        global_embd = self.global_embedding(ctxt_embd_dict, ctxt_dict)
        ctxt_mask = ctxt_dict["ctxt_mask"]
        ctxt_embd = ctxt_embd_dict["ctxt_embd"]

        fs_embd = fs_embd * fs_mask.unsqueeze(-1)
        ctxt_embd = ctxt_embd * ctxt_mask.unsqueeze(-1)

        if self.attn_type == "flex":
            B, S = fs_mask.size()
            attn_mask_ctxt = create_block_mask(
                self.create_padding_mask(ctxt_mask),
                B=B,
                H=None,
                Q_LEN=S,
                KV_LEN=S,
                _compile=True,
            )
            attn_mask_fs = create_block_mask(
                self.create_padding_mask(fs_mask),
                B=B,
                H=None,
                Q_LEN=S,
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
