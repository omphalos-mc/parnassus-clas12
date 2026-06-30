import pytest
import torch
from torch import nn

from parnassus_core.flow.modules.optimizer import Lion


def test_lion_applies_decoupled_weight_decay_and_sign_update():
    parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    parameter.grad = torch.tensor([0.2, -0.4])
    optimizer = Lion(
        [parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        weight_decay=0.1,
    )

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([0.89, -0.89]))
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"],
        torch.tensor([0.002, -0.004]),
    )


def test_lion_returns_closure_loss():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = Lion([parameter], lr=0.1)

    def closure():
        optimizer.zero_grad()
        loss = parameter.square()
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(parameter, torch.tensor(0.9))


def test_lion_reduces_a_simple_training_objective():
    parameter = nn.Parameter(torch.tensor(5.0))
    optimizer = Lion([parameter], lr=0.1)
    losses = []

    for _ in range(20):
        optimizer.zero_grad()
        loss = parameter.square()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]
    assert parameter.abs() < 5.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": -1.0}, "Invalid learning rate"),
        ({"betas": (-0.1, 0.99)}, "beta parameter at index 0"),
        ({"betas": (0.9, 1.0)}, "beta parameter at index 1"),
    ],
)
def test_lion_rejects_invalid_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Lion([nn.Parameter(torch.tensor(1.0))], **kwargs)
