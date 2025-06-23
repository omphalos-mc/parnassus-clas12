from contextlib import nullcontext
import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask

from parnassus_core.flow.modules import (
    TimestepEmbedder,
    CATransformer,
    DenseNetwork,
    NamedNetwork,
    GlobalContextEmbedding,
)


class ParticleNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.time_embedding = TimestepEmbedder(**config["time_embedding"])

        self.fs_init_net = DenseNetwork(**config["fs_init_net"])

        self.ctxt_networks = nn.ModuleList()
        for net_config in config["ctxt_networks"]:
            self.ctxt_networks.append(NamedNetwork(**net_config))

        self.global_embedding = GlobalContextEmbedding(
            input_names=config["global_embedding"]["input_names"]
        )

        self.encoder = CATransformer(**config["encoder"])

        self.attn_type = self.encoder.attn_type

        if "flash" in self.attn_type:
            self.context_manager = torch.amp.autocast(
                enabled=True, dtype=torch.bfloat16, device_type="cuda"
            )
        else:
            self.context_manager = nullcontext()
        self.fs_out_net = DenseNetwork(**config["fs_out_net"])

    def create_padding_mask(self, pads_kv):
        def padding(b, h, q_idx, kv_idx):
            return pads_kv[b, kv_idx]

        return padding

    def forward(
        self,
        fs_data: torch.Tensor,
        timestep: torch.Tensor,
        ctxt_dict: dict[str, torch.Tensor],
    ):
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
            B, S = fs_mask.shape
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
            fs_embd, _ = self.encoder(
                seq_a=fs_embd,
                seq_b=ctxt_embd,
                ctxt=global_embd,
                mask_a=attn_mask_fs,
                mask_b=attn_mask_ctxt,
            )

        fs_out = self.fs_out_net(fs_embd, ctxt=global_embd)
        return fs_out
