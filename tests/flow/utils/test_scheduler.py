import pytest
import torch

from parnassus_core.flow.utils.scheduler import (
    CosineAnnealingWarmupRestarts,
    WarmupDecayConstLR,
)


def make_optimizer(lr=1.0):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    return torch.optim.SGD([parameter], lr=lr)


def step_lrs(optimizer, scheduler, count):
    values = []
    for _ in range(count):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    return values


def test_cosine_scheduler_warms_up_decays_and_restarts_with_gamma():
    optimizer = make_optimizer()
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=6,
        warmup_steps=2,
        max_lr=1.0,
        min_lr=0.1,
        gamma=0.5,
    )

    values = step_lrs(optimizer, scheduler, 8)

    assert optimizer.defaults["lr"] == 1.0
    assert values[:2] == pytest.approx([0.55, 1.0])
    assert values[2:6] == pytest.approx([0.8681980515, 0.55, 0.2318019485, 0.1])
    assert values[6:8] == pytest.approx([0.3, 0.5])
    assert scheduler.cycle == 1
    assert scheduler.max_lr == pytest.approx(0.5)


def test_cosine_scheduler_rejects_warmup_as_long_as_cycle():
    optimizer = make_optimizer()

    with pytest.raises(AssertionError):
        CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=4,
            warmup_steps=4,
        )


def test_warmup_decay_scheduler_reaches_base_lr_then_minimum():
    optimizer = make_optimizer(lr=1.0)
    scheduler = WarmupDecayConstLR(
        optimizer,
        warmup_steps=2,
        decay_steps=2,
        min_lr=0.1,
    )

    values = step_lrs(optimizer, scheduler, 5)

    assert values == pytest.approx([1.0, 0.55, 0.1, 0.1, 0.1])
