import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask

from parnassus_core.flow.modules import (
    TimestepEmbedder,
    DenseNetwork,
    MLPBlock,
    NamedNetwork,
    GlobalContextEmbedding,
)


class EventNetwork(nn.Module):
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
        self.encoder = nn.Sequential(
            *[
                MLPBlock(**config["encoder"]["net_kwargs"])
                for _ in range(config["encoder"]["num_layers"])
            ]
        )
        hidden_dim = config["fs_init_net"]["output_dim"]
        global_embd_dim = len(config["global_embedding"]["input_names"]) * hidden_dim
        self.fs_out_net = DenseNetwork(**config["fs_out_net"], ctxt_dim=global_embd_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_embd_dim, hidden_dim * 2, bias=True),
        )

    def forward(
        self,
        fs_data: torch.Tensor,
        timestep: torch.Tensor,
        ctxt_dict: dict[str, torch.Tensor],
    ):
        fs_embd = self.fs_init_net(fs_data)

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
        fs_embd = (1 + scale_mlp) * fs_embd + shift_mlp
        fs_embd = self.encoder(fs_embd)
        fs_out = self.fs_out_net(fs_embd, ctxt=global_embd)
        return fs_out
