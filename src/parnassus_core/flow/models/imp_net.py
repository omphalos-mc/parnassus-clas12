from contextlib import nullcontext
import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask

from parnassus_core.flow.modules import (
    TimestepEmbedder,
    MLPBlock,
    DenseNetwork,
    NamedNetwork,
    GlobalContextEmbedding,
)
from parnassus_core.flow.modules.transformer import CALayer


class ImpactNetwork(nn.Module):
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

        if "transformer" in config.keys():
            self.transformer = nn.ModuleList(
                [
                    CALayer(
                        **config["transformer"]["net_kwargs"],
                    )
                    for _ in range(config["encoder"]["num_layers"])
                ]
            )

            self.attn_type = self.transformer[0].attn_type

            if "flash" in self.attn_type:
                self.context_manager = torch.amp.autocast(
                    enabled=True, dtype=torch.bfloat16, device_type="cuda"
                )
            else:
                self.context_manager = nullcontext()
        if "resnet" in config.keys():
            self.resnet = nn.ModuleList(
                [
                    MLPBlock(**config["resnet"]["net_kwargs"])
                    for _ in range(config["resnet"]["num_layers"])
                ]
            )

        hidden_dim = config["fs_init_net"]["output_dim"]
        global_embd_dim = len(config["global_embedding"]["input_names"]) * hidden_dim
        self.fs_out_net = DenseNetwork(**config["fs_out_net"], ctxt_dim=global_embd_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_embd_dim, hidden_dim * 2, bias=True),
        )

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
        fs_embd = self.fs_init_net(fs_data, ctxt=ctxt_dict.get("pf_ctxt_data", None))

        ctxt_embd_dict = {}
        ctxt_net: NamedNetwork
        for ctxt_net in self.ctxt_networks:
            ctxt_embd_dict[ctxt_net.input_name + "_embd"] = ctxt_net(ctxt_dict)

        ctxt_embd_dict["time_embd"] = self.time_embedding(timestep)

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

        if hasattr(self, "transformer"):
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
            else:
                attn_mask_ctxt = ctxt_mask
            with self.context_manager:
                layer: CALayer
                for layer in self.transformer:
                    fs_embd = layer(
                        seq_q=fs_embd,
                        seq_k=ctxt_embd,
                        ctxt=global_embd,
                        mask_k=attn_mask_ctxt,
                    )
        if hasattr(self, "resnet"):
            for block in self.resnet:
                fs_embd = block(fs_embd)

        fs_out = self.fs_out_net(fs_embd, ctxt=global_embd)
        return fs_out
