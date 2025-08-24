import torch
import pytest


from parnassus_core.flow.modules import (
    TimestepEmbedder,
    DenseNetwork,
    NamedNetwork,
    GlobalContextEmbedding,
    DiTEncoder,
)

from parnassus_core.flow.models.evt_net import EventNetwork
from parnassus_core.flow.models.part_net import ParticleNetwork
from parnassus_core.flow.models.imp_net import ImpactNetwork

torch.manual_seed(42)

HAS_GPU = torch.cuda.is_available()
ATTN_TYPES_GPU = {"flex", "flash", "flash-varlen"}
DEVICE = "cuda" if HAS_GPU else "cpu"


BS = 4
SEQ_LEN = 16

ACT = "GELU"
FS_DIM = 4
CTXT_DIM = 12
CTXT_GLOBAL_DIM = 10
HIDDEN_DIM = 128


@pytest.fixture(params=["event", "particle"])
def input_data(request):
    if request.param == "event":
        fs_shape = (BS, FS_DIM)
    else:
        fs_shape = (BS, SEQ_LEN, FS_DIM)
    fs_data = torch.randn(fs_shape, dtype=torch.float16, device=DEVICE)
    ctxt_data = torch.randn(BS, SEQ_LEN, CTXT_DIM, dtype=torch.float16, device=DEVICE)
    ctxt_global_data = torch.randn(
        BS, CTXT_GLOBAL_DIM, dtype=torch.float16, device=DEVICE
    )
    timestep = torch.rand((BS,), dtype=torch.float16, device=DEVICE)
    ctxt_mask = torch.ones(BS, SEQ_LEN, device=DEVICE).bool()
    ctxt_mask[:, SEQ_LEN // 2 :] = False
    ctxt_dict = {
        "ctxt_data": ctxt_data,
        "ctxt_mask": ctxt_mask,
        "ctxt_global_data": ctxt_global_data,
    }
    if request.param == "particle":
        ctxt_dict["fs_mask"] = torch.ones(BS, SEQ_LEN, device=DEVICE).bool()
        ctxt_dict["fs_mask"][:, SEQ_LEN // 2 :] = False
    return fs_data, timestep, ctxt_dict


@pytest.fixture
def time_embedder():
    return TimestepEmbedder(hidden_dim=HIDDEN_DIM, time_factor=1000)


@pytest.fixture
def fs_init_net():
    return DenseNetwork(
        input_dim=FS_DIM,
        output_dim=HIDDEN_DIM,
        act_h=ACT,
    )


@pytest.fixture
def ctxt_networks():
    return torch.nn.ModuleList(
        [
            NamedNetwork(
                input_name="ctxt",
                fields=["data"],
                dense_kwargs={
                    "input_dim": CTXT_DIM,
                    "output_dim": HIDDEN_DIM,
                    "act_h": ACT,
                },
            ),
            NamedNetwork(
                input_name="ctxt_global",
                fields=["data"],
                dense_kwargs={
                    "input_dim": CTXT_GLOBAL_DIM,
                    "output_dim": HIDDEN_DIM,
                    "act_h": ACT,
                },
            ),
        ]
    )


@pytest.fixture
def global_embedding():
    return GlobalContextEmbedding(
        input_names=["ctxt", "ctxt_global", "time"],
    )


@pytest.fixture
def transformer_encoder():
    return DiTEncoder(
        hidden_dim=HIDDEN_DIM,
        num_heads=8,
        num_layers=2,
        attn_type="torch",
        bidirectional=True,
    )


@pytest.fixture
def dense_encoder():
    return DenseNetwork(
        input_dim=HIDDEN_DIM,
        output_dim=HIDDEN_DIM,
    )


@pytest.fixture
def fs_out_net():
    return DenseNetwork(
        input_dim=HIDDEN_DIM,
        output_dim=FS_DIM,
        ctxt_dim=HIDDEN_DIM * 3,
        act_h=ACT,
    )


class TestEventNetwork:
    @pytest.fixture
    def event_network(
        self,
        time_embedder,
        fs_init_net,
        ctxt_networks,
        global_embedding,
        dense_encoder,
        fs_out_net,
    ):
        return (
            EventNetwork(
                time_embedding=time_embedder,
                fs_init_net=fs_init_net,
                ctxt_networks=ctxt_networks,
                global_embedding=global_embedding,
                encoder=dense_encoder,
                fs_out_net=fs_out_net,
            )
            .to(DEVICE)
            .half()
        )

    @pytest.mark.parametrize("input_data", ["event"], indirect=True)
    def test_evt_network(self, input_data, event_network):
        fs_data, timestep, ctxt_dict = input_data
        output = event_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape


class TestParticleNetwork:
    @pytest.fixture
    def particle_network(
        self,
        time_embedder,
        fs_init_net,
        ctxt_networks,
        global_embedding,
        transformer_encoder,
        fs_out_net,
    ):
        return (
            ParticleNetwork(
                time_embedding=time_embedder,
                fs_init_net=fs_init_net,
                ctxt_networks=ctxt_networks,
                global_embedding=global_embedding,
                encoder=transformer_encoder,
                fs_out_net=fs_out_net,
            )
            .to(DEVICE)
            .half()
        )

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    def test_particle_network(
        self,
        input_data,
        particle_network,
    ):
        fs_data, timestep, ctxt_dict = input_data
        output = particle_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    @pytest.mark.parametrize("attn_type", ["torch", "flex", "flash-varlen"])
    @pytest.mark.parametrize("attn_type_new", ["torch", "flex", "flash-varlen"])
    def test_change_backend(
        self,
        input_data,
        attn_type,
        attn_type_new,
        particle_network,
    ):
        if not HAS_GPU and (
            attn_type in ATTN_TYPES_GPU or attn_type_new in ATTN_TYPES_GPU
        ):
            pytest.skip("Skipping GPU-specific test on CPU-only environment")
        fs_data, timestep, ctxt_dict = input_data
        particle_network.change_backend(attn_type)
        output = particle_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        particle_network.change_backend(attn_type_new)
        output_new = particle_network(fs_data, timestep, ctxt_dict)
        assert output_new.shape == fs_data.shape

        assert torch.allclose(output, output_new, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    @pytest.mark.parametrize("attn_type", ["torch", "flex", "flash-varlen"])
    def test_masking(
        self,
        input_data,
        attn_type,
        particle_network,
    ):
        if not HAS_GPU and (attn_type in ATTN_TYPES_GPU):
            pytest.skip("Skipping GPU-specific test on CPU-only environment")
        fs_data, timestep, ctxt_dict = input_data
        fs_mask = ctxt_dict["fs_mask"]
        particle_network.change_backend(attn_type)
        output = particle_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        fs_data[~fs_mask] *= 1e3

        output_new = particle_network(fs_data, timestep, ctxt_dict)
        assert output_new.shape == fs_data.shape

        assert torch.allclose(
            output[fs_mask],
            output_new[fs_mask],
            atol=1e-3,
            rtol=1e-3,
        )


class TestImpactNetwork:
    @pytest.fixture
    def impact_network(
        self,
        time_embedder,
        fs_init_net,
        ctxt_networks,
        global_embedding,
        transformer_encoder,
        dense_encoder,
        fs_out_net,
    ):
        return (
            ImpactNetwork(
                time_embedding=time_embedder,
                fs_init_net=fs_init_net,
                ctxt_networks=ctxt_networks,
                global_embedding=global_embedding,
                fs_out_net=fs_out_net,
                encoder=transformer_encoder,
                resnet=dense_encoder,
            )
            .to(DEVICE)
            .half()
        )

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    def test_impact_network(
        self,
        input_data,
        impact_network,
    ):
        fs_data, timestep, ctxt_dict = input_data
        output = impact_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    @pytest.mark.parametrize("attn_type", ["torch", "flex", "flash-varlen"])
    @pytest.mark.parametrize("attn_type_new", ["torch", "flex", "flash-varlen"])
    def test_change_backend(
        self,
        input_data,
        attn_type,
        attn_type_new,
        impact_network,
    ):
        if not HAS_GPU and (
            attn_type in ATTN_TYPES_GPU or attn_type_new in ATTN_TYPES_GPU
        ):
            pytest.skip("Skipping GPU-specific test on CPU-only environment")
        fs_data, timestep, ctxt_dict = input_data
        impact_network.change_backend(attn_type)
        output = impact_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        impact_network.change_backend(attn_type_new)
        output_new = impact_network(fs_data, timestep, ctxt_dict)
        assert output_new.shape == fs_data.shape

        assert torch.allclose(output, output_new, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    @pytest.mark.parametrize("attn_type", ["torch", "flex", "flash-varlen"])
    def test_masking(
        self,
        input_data,
        attn_type,
        impact_network,
    ):
        if not HAS_GPU and (attn_type in ATTN_TYPES_GPU):
            pytest.skip("Skipping GPU-specific test on CPU-only environment")
        fs_data, timestep, ctxt_dict = input_data
        fs_mask = ctxt_dict["fs_mask"]
        impact_network.change_backend(attn_type)
        output = impact_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        fs_data[~fs_mask] *= 1e3

        output_new = impact_network(fs_data, timestep, ctxt_dict)
        assert output_new.shape == fs_data.shape

        assert torch.allclose(
            output[fs_mask],
            output_new[fs_mask],
            atol=1e-3,
            rtol=1e-3,
        )
