from typing import Literal
from .dpm import DPM_Solver, NoiseScheduleFlow
from .sampler import EulerSampler, HeunSampler, PNDMSampler, SDESampler

SAMPLERS = {
    "dpm": DPM_Solver,
    "euler": EulerSampler,
    "sde": SDESampler,
    "pndm": PNDMSampler,
    "heun": HeunSampler,
}

SamplerType = Literal["dpm", "euler", "sde", "pndm", "heun"]

__all__ = [
    "SAMPLERS",
    "SamplerType",
    "DPM_Solver",
    "NoiseScheduleFlow",
    "EulerSampler",
    "SDESampler",
    "HeunSampler",
    "PNDMSampler",
]
