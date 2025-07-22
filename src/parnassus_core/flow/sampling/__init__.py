from typing import Literal
from .dpm import DPM_Solver, NoiseScheduleFlow
from .sampler import EulerSampler, HeunSampler, PNDMSampler

SAMPLERS = {
    "dpm": DPM_Solver,
    "euler": EulerSampler,
    "pndm": PNDMSampler,
    "heun": HeunSampler,
}

SamplerType = Literal["dpm", "euler", "pndm", "heun"]

__all__ = [
    "SAMPLERS",
    "SamplerType",
    "DPM_Solver",
    "NoiseScheduleFlow",
    "EulerSampler",
    "HeunSampler",
    "PNDMSampler",
]
