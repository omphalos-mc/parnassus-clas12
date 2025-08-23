import torch
from torch import nn

from parnassus_core.flow.modules.optimizer import Lion


def test_lion():
    model = nn.Linear(10, 10)
    optimizer = Lion(model.parameters(), lr=0.01, weight_decay=0.01)

    x = torch.randn(5, 10)
    y = torch.randn(5, 10)

    criterion = nn.MSELoss()

    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1.0, "Lion optimizer did not reduce the loss sufficiently."
