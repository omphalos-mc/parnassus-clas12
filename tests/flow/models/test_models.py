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
        output, guidance = event_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape
        assert guidance is None


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

    @pytest.fixture
    def particle_network_ctxt_as_token(
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
                ctxt_as_token=True,
                num_time_tokens=2,
                num_ctxt_tokens=3,
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
        output, guidance = particle_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape
        assert guidance is None

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    @pytest.mark.parametrize("attn_type", ["torch", "flex"])
    def test_particle_network_ctxt_as_token(
        self,
        input_data,
        attn_type,
        particle_network_ctxt_as_token,
    ):
        if not HAS_GPU and attn_type in ATTN_TYPES_GPU:
            pytest.skip("Skipping GPU-specific test on CPU-only environment")
        fs_data, timestep, ctxt_dict = input_data
        particle_network_ctxt_as_token.change_backend(attn_type)
        output, guidance = particle_network_ctxt_as_token(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape
        assert guidance is None

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    def test_append_tokens_to_ctxt_seq(
        self,
        input_data,
        particle_network_ctxt_as_token,
    ):
        _, timestep, ctxt_dict = input_data
        ctxt_embd_dict = {}
        for ctxt_net in particle_network_ctxt_as_token.ctxt_networks:
            ctxt_embd_dict[ctxt_net.input_name + "_embd"] = ctxt_net(ctxt_dict)
        ctxt_embd_dict["time_embd"] = particle_network_ctxt_as_token.time_embedding(
            timestep
        )
        ctxt_seq = ctxt_embd_dict["ctxt_embd"]
        ctxt_mask = ctxt_dict["ctxt_mask"]

        ctxt_seq_with_tokens, ctxt_mask_with_tokens = (
            particle_network_ctxt_as_token.append_tokens_to_ctxt_seq(
                ctxt_seq=ctxt_seq,
                ctxt_embd_dict=ctxt_embd_dict,
                ctxt_mask=ctxt_mask,
            )
        )

        total_tokens = particle_network_ctxt_as_token.total_tokens
        assert ctxt_seq_with_tokens.shape == (
            ctxt_seq.size(0),
            ctxt_seq.size(1) + total_tokens,
            ctxt_seq.size(2),
        )
        assert ctxt_mask_with_tokens.shape == (
            ctxt_mask.size(0),
            ctxt_mask.size(1) + total_tokens,
        )
        assert torch.all(ctxt_mask_with_tokens[:, :total_tokens])
        torch.testing.assert_close(ctxt_mask_with_tokens[:, total_tokens:], ctxt_mask)
        torch.testing.assert_close(ctxt_seq_with_tokens[:, total_tokens:], ctxt_seq)

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
        output, _ = particle_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        particle_network.change_backend(attn_type_new)
        output_new, _ = particle_network(fs_data, timestep, ctxt_dict)
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
        output, _ = particle_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        perturbed = fs_data.clone()
        perturbed[~fs_mask] *= 1e3

        output_new, _ = particle_network(perturbed, timestep, ctxt_dict)
        assert output_new.shape == fs_data.shape

        assert torch.allclose(
            output[fs_mask],
            output_new[fs_mask],
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("input_data", ["particle"], indirect=True)
    @pytest.mark.parametrize("attn_type", ["torch", "flex"])
    def test_padding_invariance(
        self,
        input_data,
        attn_type,
        particle_network,
    ):
        """Slicing off slots every event pads leaves the real outputs unchanged.

        ``EvaluationModel`` with ``trim_padding`` relies on this to hand the
        network only the slots some event in the batch fills.
        """
        if not HAS_GPU and attn_type in ATTN_TYPES_GPU:
            pytest.skip("Skipping GPU-specific test on CPU-only environment")
        fs_data, timestep, ctxt_dict = input_data
        # fp32, so a real mismatch cannot hide behind fp16 rounding.
        particle_network.float().change_backend(attn_type)
        fs_data, timestep = fs_data.float(), timestep.float()
        slots = torch.arange(SEQ_LEN, device=DEVICE)
        fs_counts = torch.tensor([1, 5, 8, 3], device=DEVICE)
        ctxt_counts = torch.tensor([2, 8, 6, 4], device=DEVICE)
        ctxt_dict = {
            "ctxt_data": ctxt_dict["ctxt_data"].float(),
            "ctxt_global_data": ctxt_dict["ctxt_global_data"].float(),
            "ctxt_mask": slots < ctxt_counts[:, None],
            "fs_mask": slots < fs_counts[:, None],
        }
        fs_len, ctxt_len = int(fs_counts.max()), int(ctxt_counts.max())
        trimmed = ctxt_dict | {
            "ctxt_data": ctxt_dict["ctxt_data"][:, :ctxt_len],
            "ctxt_mask": ctxt_dict["ctxt_mask"][:, :ctxt_len],
            "fs_mask": ctxt_dict["fs_mask"][:, :fs_len],
        }

        output, _ = particle_network(fs_data, timestep, ctxt_dict)
        output_trimmed, _ = particle_network(fs_data[:, :fs_len], timestep, trimmed)

        fs_mask = trimmed["fs_mask"]
        torch.testing.assert_close(output_trimmed[fs_mask], output[:, :fs_len][fs_mask])


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
        output, guidance = impact_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape
        assert guidance is None

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
        output, _ = impact_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        impact_network.change_backend(attn_type_new)
        output_new, _ = impact_network(fs_data, timestep, ctxt_dict)
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
        output, _ = impact_network(fs_data, timestep, ctxt_dict)
        assert output.shape == fs_data.shape

        perturbed = fs_data.clone()
        perturbed[~fs_mask] *= 1e3

        output_new, _ = impact_network(perturbed, timestep, ctxt_dict)
        assert output_new.shape == fs_data.shape

        assert torch.allclose(
            output[fs_mask],
            output_new[fs_mask],
            atol=1e-3,
            rtol=1e-3,
        )
