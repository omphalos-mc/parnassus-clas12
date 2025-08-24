import torch
import pytest
import torch.nn.functional as F

from parnassus_core.flow.utils.loss import MSELoss, MSEAndDirectionLoss, Set2SetLoss

torch.manual_seed(42)

HAS_GPU = torch.cuda.is_available()
ATTN_TYPES_GPU = {"flex", "flash", "flash-varlen"}
DEVICE = "cuda" if HAS_GPU else "cpu"

BS = 2
SEQ_LEN = 8

ACT = "GELU"
DIM = 4


@pytest.fixture(params=["event", "particle"])
def input_data(request):
    if request.param == "event":
        shape = (BS, DIM)
    else:
        shape = (BS, SEQ_LEN, DIM)
    pred = torch.ones(shape, device=DEVICE)
    target = torch.zeros(shape, device=DEVICE)
    if request.param == "particle":
        mask = torch.ones(BS, SEQ_LEN, device=DEVICE).bool()
        mask[:, SEQ_LEN // 2 :] = False
    else:
        mask = None
    return pred, target, mask


class TestMSELoss:
    def test_wrong_reduction(self):
        with pytest.raises(AssertionError):
            _ = MSELoss(reduction="invalid").to(DEVICE)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_forward(self, input_data, reduction):
        pred, target, mask = input_data
        criterion = MSELoss(reduction=reduction).to(DEVICE)
        loss = criterion(pred, target, mask)
        if mask is None:
            mask = torch.ones_like(pred[..., 0]).bool()
        expected_loss = F.mse_loss(
            pred[mask],
            target[mask],
            reduction=reduction,
        )
        torch.testing.assert_close(loss, expected_loss)


class TestMSEAndDirectionLoss:
    def test_wrong_reduction(self):
        with pytest.raises(AssertionError):
            _ = MSEAndDirectionLoss(reduction="invalid").to(DEVICE)

    def test_wrong_cosine_dim(self):
        with pytest.raises(AssertionError):
            _ = MSEAndDirectionLoss(cosine_sim_dim=0).to(DEVICE)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_mse_and_direction_loss(self, input_data, reduction):
        from parnassus_core.flow.utils.loss import MSEAndDirectionLoss

        pred, target, mask = input_data

        criterion = MSEAndDirectionLoss(
            cosine_sim_dim=2 if pred.dim() > 2 else 1,
            reduction=reduction,
        ).to(DEVICE)
        loss = criterion(pred, target, mask)
        if mask is None:
            mask = torch.ones_like(pred[..., 0]).bool()
        mse_loss = F.mse_loss(
            pred[mask],
            target[mask],
            reduction=reduction,
        )
        direction_loss = 1.0 - F.cosine_similarity(pred[mask], target[mask], dim=1)
        direction_loss = (
            direction_loss.mean() if reduction == "mean" else direction_loss.sum()
        )
        expected_loss = mse_loss + direction_loss
        torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize("input_data", ["particle"], indirect=True)
def test_set2set_loss(input_data):
    pred, target, mask = input_data

    criterion = Set2SetLoss().to(DEVICE)
    _ = criterion(pred, target, mask)
