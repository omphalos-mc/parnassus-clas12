from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from parnassus_core.flow.modules import (
    NamedNetwork,
)


class ImpactNetwork(nn.Module):
    def __init__(
        self,
        time_embedding: nn.Module,
        fs_init_net: nn.Module,
        ctxt_networks: nn.ModuleList,
        global_embedding: nn.Module,
        fs_out_net: nn.Module,
        fourier_embedding: nn.Module | None = None,
        encoder: nn.Module | None = None,
        resnet: nn.Module | None = None,
    ):
        super().__init__()
        self.time_shape = (-1, 1, 1)  # (batch_size, 1, 1)

        self.time_embedding = time_embedding
        self.fs_init_net = fs_init_net
        self.ctxt_networks = ctxt_networks
        self.global_embedding = global_embedding
        self.fs_out_net = fs_out_net
        self.encoder = encoder
        self.resnet = resnet
        self.fourier_embedding = fourier_embedding

        self.context_manager = nullcontext()
        self.attn_type = (
            self.change_backend(encoder.attn_type) if encoder is not None else ""
        )

        hidden_dim = self.time_embedding.hidden_dim
        global_embd_dim = len(self.global_embedding.input_names) * hidden_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_embd_dim, hidden_dim * 2, bias=True),
        )

    def create_padding_mask(self, pads_kv):
        def padding(b, h, q_idx, kv_idx):
            return pads_kv[b, kv_idx]

        return padding

    def change_backend(self, attn_type: str):
        if self.encoder is not None:
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
    ):
        ctxt_embd_dict: dict[str, torch.Tensor] = {}
        ctxt_net: NamedNetwork
        for ctxt_net in self.ctxt_networks:
            ctxt_embd_dict[ctxt_net.input_name + "_embd"] = ctxt_net(ctxt_dict)
        ctxt_embd_dict["time_embd"] = self.time_embedding(timestep)

        fs_mask = ctxt_dict["fs_mask"]
        if self.fourier_embedding is not None:
            fourier = self.fourier_embedding(fs_data)
            fs_data = torch.cat([fs_data, fourier], dim=-1)

        fs_embd = self.fs_init_net(fs_data)
        if "pf_ctxt_embd" in ctxt_embd_dict:
            fs_embd = torch.cat([fs_embd, ctxt_embd_dict["pf_ctxt_embd"]], dim=-1)

        global_embd = self.global_embedding(ctxt_embd_dict, ctxt_dict)
        (
            shift_mlp,
            scale_mlp,
        ) = self.adaLN_modulation(global_embd).chunk(2, dim=-1)

        scale_mlp = scale_mlp.view(-1, 1, fs_embd.shape[-1])
        shift_mlp = shift_mlp.view(-1, 1, fs_embd.shape[-1])
        fs_embd = (1 + scale_mlp) * fs_embd + shift_mlp

        fs_embd = fs_embd * fs_mask.unsqueeze(-1)
        if "ctxt_embd" in ctxt_embd_dict:
            ctxt_mask = ctxt_dict["ctxt_mask"]
            ctxt_embd = ctxt_embd_dict["ctxt_embd"]
            ctxt_embd = ctxt_embd * ctxt_mask.unsqueeze(-1)

        if self.encoder is not None:
            if self.attn_type == "flex":
                B, S = fs_mask.shape
                attn_mask_fs = create_block_mask(
                    self.create_padding_mask(fs_mask),
                    B=B,
                    H=None,
                    Q_LEN=S,
                    KV_LEN=S,
                    _compile=True,
                )
                attn_mask_ctxt = create_block_mask(
                    self.create_padding_mask(ctxt_mask),
                    B=B,
                    H=None,
                    Q_LEN=S,
                    KV_LEN=S,
                    _compile=True,
                )
            else:
                attn_mask_fs = fs_mask
                attn_mask_ctxt = ctxt_mask
            with self.context_manager:
                fs_embd, _ = self.encoder(
                    seq_a=fs_embd,
                    seq_b=ctxt_embd,
                    ctxt=global_embd,
                    mask_a=attn_mask_fs,
                    mask_b=attn_mask_ctxt,
                )
        if self.resnet is not None:
            fs_embd = self.resnet(fs_embd.to(global_embd))

        fs_out = self.fs_out_net(fs_embd, ctxt=global_embd)
        return fs_out
