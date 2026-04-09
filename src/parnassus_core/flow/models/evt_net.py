import torch
import torch.nn as nn


from parnassus_core.flow.modules import (
    NamedNetwork,
)


class EventNetwork(nn.Module):
    def __init__(
        self,
        time_embedding: nn.Module,
        fs_init_net: nn.Module,
        ctxt_networks: nn.ModuleList,
        global_embedding: nn.Module,
        encoder: nn.Module,
        fs_out_net: nn.Module,
    ):
        super().__init__()
        self.time_shape = (-1, 1)  # (batch_size, 1)

        self.time_embedding = time_embedding
        self.fs_init_net = fs_init_net
        self.ctxt_networks = ctxt_networks
        self.global_embedding = global_embedding
        self.encoder = encoder
        self.fs_out_net = fs_out_net

        hidden_dim = self.time_embedding.hidden_dim
        global_embd_dim = len(self.global_embedding.input_names) * hidden_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_embd_dim, hidden_dim * 2, bias=True),
        )

    def change_backend(self, backend: str):
        pass  # For compatibility with ParticleNetwork

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
        return fs_out, None
