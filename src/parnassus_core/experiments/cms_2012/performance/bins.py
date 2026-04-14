import numpy as np
from collections import defaultdict

from parnassus_core.performance.bins import BinsConfig, BinsDictT
from .config import CMS2012Sample


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
DEFAULT_EVENT_BINS["npart"].update(
    {
        CMS2012Sample.TTBAR: np.arange(0, 400, 10) + 0.5,
    }
)
DEFAULT_EVENT_BINS["ht"].update(
    {
        CMS2012Sample.TTBAR: np.linspace(0, 1000, 50),
        CMS2012Sample.QCD15TO3000: np.linspace(0, 2000, 100),
        CMS2012Sample.DYJETS: np.linspace(0, 500, 50),
    }
)

DEFAULT_EVENT_RES_BINS = {
    "npart_res": defaultdict(constant_factory(np.arange(-200, 200, 8) + 0.5)),
    "njet_res": defaultdict(constant_factory(np.arange(-4, 4, 1) + 0.5)),
    "ht_res": defaultdict(constant_factory(np.linspace(-0.4, 1.0, 50))),
    "met_x_res": defaultdict(constant_factory(np.linspace(-2, 2, 50))),
    "met_y_res": defaultdict(constant_factory(np.linspace(-2, 2, 50))),
}
DEFAULT_EVENT_RES_BINS["ht_res"].update(
    {
        CMS2012Sample.DYJETS: np.linspace(-0.5, 4.0, 50),
    }
)
DEFAULT_EVENT_RES_BINS["met_x_res"].update(
    {
        CMS2012Sample.TTBAR: np.linspace(-2, 2, 50),
        CMS2012Sample.QCD15TO3000: np.linspace(-5, 5, 50),
        CMS2012Sample.DYJETS: np.linspace(-5, 5, 50),
    }
)
DEFAULT_EVENT_RES_BINS["met_y_res"].update(
    {
        CMS2012Sample.TTBAR: np.linspace(-2, 2, 50),
        CMS2012Sample.QCD15TO3000: np.linspace(-5, 5, 50),
        CMS2012Sample.DYJETS: np.linspace(-5, 5, 50),
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
        CMS2012Sample.TTBAR: np.linspace(10, 200, 100),
        CMS2012Sample.QCD15TO3000: np.linspace(0, 800, 100),
        CMS2012Sample.DYJETS: np.linspace(0, 400, 100),
    }
)

DEFAULT_JET_RES_BINS = {
    "pt_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "eta_res": defaultdict(constant_factory(np.linspace(-0.05, 0.05, 50))),
    "phi_res": defaultdict(constant_factory(np.linspace(-0.05, 0.05, 50))),
    "d2_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "c2_res": defaultdict(constant_factory(np.linspace(-0.2, 0.2, 50))),
}

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


class CMS2012BinsConfig(BinsConfig[CMS2012Sample]):
    def __init__(
        self,
        event_bins: BinsDictT | None = None,
        event_res_bins: BinsDictT | None = None,
        jet_bins: BinsDictT | None = None,
        jet_res_bins: BinsDictT | None = None,
        particle_bins: BinsDictT | None = None,
        particle_res_bins: BinsDictT | None = None,
    ):
        super().__init__(
            event_bins=dict(DEFAULT_EVENT_BINS) if event_bins is None else event_bins,
            event_res_bins=dict(DEFAULT_EVENT_RES_BINS)
            if event_res_bins is None
            else event_res_bins,
            jet_bins=dict(DEFAULT_JET_BINS) if jet_bins is None else jet_bins,
            jet_res_bins=dict(DEFAULT_JET_RES_BINS)
            if jet_res_bins is None
            else jet_res_bins,
            particle_bins=dict(DEFAULT_PARTICLE_BINS)
            if particle_bins is None
            else particle_bins,
            particle_res_bins=dict(DEFAULT_PARTICLE_RES_BINS)
            if particle_res_bins is None
            else particle_res_bins,
        )
