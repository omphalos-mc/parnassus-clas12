import math
from dataclasses import dataclass
from typing import Iterator

import torch

from parnassus_core.flow.utils.probability_path import (
    FlowType,
    ProbabilityPath,
    make_probability_path,
)


@dataclass(frozen=True)
class FlowSample:
    time: torch.Tensor
    state: torch.Tensor
    velocity: torch.Tensor
    source: torch.Tensor
    data: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Preserve the previous t, x_t, u_t unpacking API."""
        yield self.time
        yield self.state
        yield self.velocity

    def target(self, target_type: str) -> torch.Tensor:
        if target_type == "velocity":
            return self.velocity
        if target_type == "data":
            return self.data
        if target_type == "noise":
            return self.source
        raise ValueError(f"Unknown target type: {target_type}")


class FlowMatcher:
    def __init__(
        self,
        flow_type: FlowType = "icfm",
        time_distribution: str = "uniform",
        power_alpha: float = 3,
        lognorm_mu: float = 0,
        lognorm_sigma: float = 1,
        mode_s: float = -0.54,
    ):
        if time_distribution not in {"uniform", "pow", "lognorm", "mode"}:
            raise ValueError(f"Unknown time distribution: {time_distribution}")
        self.flow_type = flow_type
        self.path: ProbabilityPath = make_probability_path(flow_type)
        self.time_distribution = time_distribution
        self.power_alpha = power_alpha
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma
        self.mode_s = mode_s

    def sample_time(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.time_distribution == "uniform":
            return torch.rand(size, device=device, dtype=dtype)
        if self.time_distribution == "pow":
            uniform = torch.rand(size, device=device, dtype=dtype)
            return uniform.pow(1 / self.power_alpha)
        if self.time_distribution == "lognorm":
            normal = torch.randn(size, device=device, dtype=dtype)
            return torch.sigmoid(self.lognorm_mu + self.lognorm_sigma * normal)

        uniform = torch.rand(size, device=device, dtype=dtype)
        return (
            1
            - uniform
            - self.mode_s * (torch.cos(math.pi / 2 * uniform).square() - 1 + uniform)
        )

    def sample(self, data: torch.Tensor) -> FlowSample:
        source = torch.randn_like(data)
        time = self.sample_time(data.shape[0], data.device, data.dtype)
        state = self.path.interpolate(source, data, time)
        velocity = self.path.velocity(source, data, time)
        return FlowSample(time, state, velocity, source, data)
