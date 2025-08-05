from dataclasses import dataclass
from typing import Generic, TypeVar, TypeAlias
from .config import BaseSample

from numpy.typing import NDArray


def constant_factory(value):
    """Factory function to create a constant value."""
    return lambda: value


# Bins configuration for different samples and variables
T = TypeVar("T", bound=BaseSample)
BinsDictT: TypeAlias = dict[str, dict[T, NDArray]]


@dataclass(slots=True)
class BinsConfig(Generic[T]):
    event_bins: BinsDictT
    event_res_bins: BinsDictT

    jet_bins: BinsDictT
    jet_res_bins: BinsDictT

    particle_bins: BinsDictT
    particle_res_bins: BinsDictT
