import numpy as np
from collections import defaultdict

from .config import Sample


def constant_factory(value):
    """Factory function to create a constant value."""
    return lambda: value


# Event bins configuration

DEFAULT_EVENT_BINS = {
    "npart": defaultdict(constant_factory(np.arange(0, 300, 10) + 0.5)),
    "njet": defaultdict(constant_factory(np.arange(-1, 10, 1) + 0.5)),
    "ht": defaultdict(constant_factory(np.linspace(0, 1000, 100))),
    "met_x": defaultdict(constant_factory(np.linspace(-80, 80, 50))),
    "met_y": defaultdict(constant_factory(np.linspace(-80, 80, 50))),
}
DEFAULT_EVENT_BINS["ht"].update(
    {
        Sample.H4LEP: np.linspace(0, 500, 50),
        Sample.TTBAR: np.linspace(0, 800, 50),
        Sample.QCD_470: np.linspace(800, 1800, 50),
        Sample.QCD_600: np.linspace(1000, 2000, 50),
    }
)

DEFAULT_EVENT_RES_BINS = {
    "npart_res": defaultdict(constant_factory(np.arange(-200, 100, 8) + 0.5)),
    "njet_res": defaultdict(constant_factory(np.arange(-3, 3, 1) + 0.5)),
    "ht_res": defaultdict(constant_factory(np.linspace(-1, 1, 50))),
    "met_x_res": defaultdict(constant_factory(np.linspace(-2, 2, 50))),
    "met_y_res": defaultdict(constant_factory(np.linspace(-2, 2, 50))),
}
DEFAULT_EVENT_RES_BINS["ht_res"].update(
    {
        Sample.H4LEP: np.linspace(-0.4, 1.0, 50),
        Sample.TTBAR: np.linspace(-0.3, 0.6, 50),
        Sample.QCD_470: np.linspace(-0.2, 0.2, 50),
        Sample.QCD_600: np.linspace(-0.2, 0.2, 50),
    }
)
DEFAULT_EVENT_RES_BINS["met_x_res"].update(
    {
        Sample.H4LEP: np.linspace(-2, 2, 50),
        Sample.TTBAR: np.linspace(-2, 2, 50),
        Sample.QCD_470: np.linspace(-5, 5, 50),
        Sample.QCD_600: np.linspace(-5, 5, 50),
    }
)
DEFAULT_EVENT_RES_BINS["met_y_res"].update(
    {
        Sample.H4LEP: np.linspace(-2, 2, 50),
        Sample.TTBAR: np.linspace(-2, 2, 50),
        Sample.QCD_470: np.linspace(-5, 5, 50),
        Sample.QCD_600: np.linspace(-5, 5, 50),
    }
)

# Jet bins configuration

DEFAULT_JET_BINS = {
    "pt": defaultdict(constant_factory(np.linspace(0, 1000, 100))),
    "eta": defaultdict(constant_factory(np.linspace(-3, 3, 50))),
    "phi": defaultdict(constant_factory(np.linspace(-np.pi, np.pi, 50))),
    "d2": defaultdict(constant_factory(np.linspace(-3, 4, 50))),
    "c2": defaultdict(constant_factory(np.linspace(0, 0.5, 50))),
}
DEFAULT_JET_BINS["pt"].update(
    {
        Sample.H4LEP: np.linspace(10, 100, 100),
        Sample.TTBAR: np.linspace(10, 200, 100),
        Sample.QCD_470: np.linspace(0, 800, 100),
        Sample.QCD_600: np.linspace(0, 900, 100),
    }
)

DEFAULT_JET_RES_BINS = {
    "pt_res": defaultdict(constant_factory(np.linspace(-0.3, 0.5, 50))),
    "eta_res": defaultdict(constant_factory(np.linspace(-0.05, 0.05, 50))),
    "phi_res": defaultdict(constant_factory(np.linspace(-0.05, 0.05, 50))),
    "d2_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "c2_res": defaultdict(constant_factory(np.linspace(-0.2, 0.2, 50))),
}
DEFAULT_JET_RES_BINS["pt_res"].update(
    {
        Sample.H4LEP: np.linspace(-0.4, 0.4, 50),
    }
)
DEFAULT_JET_RES_BINS["eta_res"].update(
    {
        Sample.QCD_470: np.linspace(-0.02, 0.02, 50),
        Sample.QCD_600: np.linspace(-0.02, 0.02, 50),
    }
)
DEFAULT_JET_RES_BINS["phi_res"].update(
    {
        Sample.QCD_470: np.linspace(-0.02, 0.02, 50),
        Sample.QCD_600: np.linspace(-0.02, 0.02, 50),
    }
)

# Particle bins configuration

DEFAULT_PARTICLE_BINS = {
    # class
    "class": defaultdict(constant_factory(np.arange(0, 6, 1) - 0.5)),
    # kinematic variables
    "pt": defaultdict(constant_factory(np.linspace(0, 200, 100))),
    "eta": defaultdict(constant_factory(np.linspace(-2.7, 2.7, 100))),
    "phi": defaultdict(constant_factory(np.linspace(-np.pi, np.pi, 100))),
    # vertex variables
    "vx": defaultdict(constant_factory(np.linspace(-2, 2, 100))),
    "vy": defaultdict(constant_factory(np.linspace(-2, 2, 100))),
    "vz": defaultdict(constant_factory(np.linspace(-20, 20, 100))),
    # impact parameter variables
    "d0": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 100))),
    "d0Error": defaultdict(constant_factory(np.linspace(0, 1, 100))),
    "d0Sig": defaultdict(constant_factory(np.linspace(-2, 2, 100))),
    "z0": defaultdict(constant_factory(np.linspace(-1, 1, 100))),
    "z0Error": defaultdict(constant_factory(np.linspace(0, 2, 100))),
    "z0Sig": defaultdict(constant_factory(np.linspace(-4, 4, 100))),
    "ip3d": defaultdict(constant_factory(np.linspace(0, 2, 100))),
    "ip3dError": defaultdict(constant_factory(np.linspace(0, 2, 100))),
    "ip3dSig": defaultdict(constant_factory(np.linspace(0, 5, 100))),
}

DEFAULT_PARTICLE_RES_BINS = {
    "pt_res": defaultdict(constant_factory(np.linspace(-1, 1, 50))),
    "eta_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "phi_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "vx_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "vy_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "vz_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
}
