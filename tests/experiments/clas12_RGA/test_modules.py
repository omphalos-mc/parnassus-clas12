import pytest
import torch

from parnassus_core.experiments.clas12_RGA.modules import SANamedNetwork
from parnassus_core.flow.modules import DiTEncoder

torch.manual_seed(42)

BS = 3
SEQ_LEN = 16
CTXT_DIM = 18
SA_DIM = 64
OUT_DIM = 96


@pytest.fixture
def sa_named_network():
    net = SANamedNetwork(
        input_name="ctxt",
        fields=["data"],
        dense_kwargs={
            "input_dim": CTXT_DIM,
            "hidden_dim": 32,
            "output_dim": SA_DIM,
            "act_h": "GELU",
        },
        sa_encoder=DiTEncoder(
            hidden_dim=SA_DIM,
            num_heads=4,
            num_layers=1,
            mlp_ratio=2.0,
            bidirectional=False,
            norm="layer",
            no_modulation=True,
        ),
        output_dim=OUT_DIM,
    )
    return net.eval()


@pytest.fixture
def ctxt_dict():
    ctxt_mask = torch.zeros(BS, SEQ_LEN, dtype=torch.bool)
    ctxt_mask[0, :5] = True
    ctxt_mask[1, :1] = True  # event 2 has no valid particles
    return {"ctxt_data": torch.randn(BS, SEQ_LEN, CTXT_DIM), "ctxt_mask": ctxt_mask}


def test_output_shape(sa_named_network, ctxt_dict):
    out = sa_named_network(ctxt_dict)
    assert out.shape == (BS, SEQ_LEN, OUT_DIM)


def test_padding_does_not_affect_valid_outputs(sa_named_network, ctxt_dict):
    mask = ctxt_dict["ctxt_mask"]
    out = sa_named_network(ctxt_dict)

    perturbed = dict(ctxt_dict)
    perturbed["ctxt_data"] = torch.where(
        mask.unsqueeze(-1), ctxt_dict["ctxt_data"], 100.0
    )
    out_perturbed = sa_named_network(perturbed)

    torch.testing.assert_close(out[mask], out_perturbed[mask])


def test_empty_event_is_finite(sa_named_network, ctxt_dict):
    out = sa_named_network(ctxt_dict)
    assert torch.isfinite(out).all()


def test_rejects_modulated_encoder():
    with pytest.raises(AssertionError):
        SANamedNetwork(
            input_name="ctxt",
            fields=["data"],
            dense_kwargs={"input_dim": CTXT_DIM, "output_dim": SA_DIM},
            sa_encoder=DiTEncoder(
                hidden_dim=SA_DIM, num_heads=4, num_layers=1, bidirectional=False
            ),
        )
