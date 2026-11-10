import numpy as np
from collections import defaultdict

from parnassus_core.performance.bins import BinsConfig, BinsDictT
from .config import Clas12RGASample  # noqa: F401


def constant_factory(value):
    """Factory function to create a constant value."""
    return lambda: value


# NOTE: These binnings are inherited from the CMS experiments and are almost certainly
# NOT tuned for CLAS12 RGA kinematics (fixed-target e-p, ~10.6 GeV beam). They are kept
# only as a working default so the plotting machinery runs out of the box. Tune the
# ranges below (or override per-sample) for CLAS12 physics.
#
# Every entry is a ``defaultdict`` keyed by ``Clas12RGASample``: the default factory
# supplies the bin edges used for any sample, and you can add per-sample overrides via
# ``DEFAULT_*_BINS[var][Clas12RGASample.CLASDIS_V0] = np.linspace(...)`` when a single
# range does not fit every sample.

# Event bins configuration

DEFAULT_EVENT_BINS = {
    "npart": defaultdict(constant_factory(np.arange(-0.5, 30.5, 1))),
    "njet": defaultdict(constant_factory(np.arange(-1, 5, 1) + 0.5)),
    "ht": defaultdict(constant_factory(np.linspace(0, 6, 100))),
    "met_x": defaultdict(constant_factory(np.linspace(-5, 5, 50))),
    "met_y": defaultdict(constant_factory(np.linspace(-5, 5, 50))),
    # p analogues (event model targets)
    "ht_p": defaultdict(constant_factory(np.linspace(0, 25, 100))),
    "Pxh": defaultdict(constant_factory(np.linspace(-4, 4, 100))),
    "Pyh": defaultdict(constant_factory(np.linspace(-4, 4, 100))),
}

DEFAULT_EVENT_RES_BINS = {
    "npart_res": defaultdict(constant_factory(np.arange(-20, 10, 1) + 0.5)),
    "njet_res": defaultdict(constant_factory(np.arange(-3, 3, 1) + 0.5)),
    "ht_res": defaultdict(constant_factory(np.linspace(-3, 4, 71))),
    "met_x_res": defaultdict(constant_factory(np.linspace(-2, 2, 50))),
    "met_y_res": defaultdict(constant_factory(np.linspace(-2, 2, 50))),
    # Every event residual is an absolute difference (reco - truth), in GeV.
    "ht_p_res": defaultdict(constant_factory(np.linspace(-12, 6, 91))),
    "Pxh_res": defaultdict(constant_factory(np.linspace(-3, 3, 61))),
    "Pyh_res": defaultdict(constant_factory(np.linspace(-3, 3, 61))),
}

# Jet bins configuration
# (Kept so the shared BinsConfig has a complete set; jet plots are not part of the
# default CLAS12 notebook but the machinery still clusters jets during processing.)

DEFAULT_JET_BINS = {
    "pt": defaultdict(constant_factory(np.linspace(0, 1000, 100))),
    "eta": defaultdict(constant_factory(np.linspace(-3, 3, 50))),
    "phi": defaultdict(constant_factory(np.linspace(-np.pi, np.pi, 50))),
    "d2": defaultdict(constant_factory(np.linspace(-3, 4, 50))),
    "c2": defaultdict(constant_factory(np.linspace(0, 0.5, 50))),
}

DEFAULT_JET_RES_BINS = {
    "pt_res": defaultdict(constant_factory(np.linspace(-0.3, 0.5, 50))),
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
    "pt": defaultdict(constant_factory(np.linspace(0, 2.5, 100))),
    "p": defaultdict(constant_factory(np.linspace(0, 10, 100))),
    "eta": defaultdict(constant_factory(np.linspace(-1, 5, 100))),
    "theta": defaultdict(constant_factory(np.linspace(0, 1.0, 100))),
    "phi": defaultdict(constant_factory(np.linspace(-np.pi, np.pi, 100))),
    # vertex variables
    "vx": defaultdict(constant_factory(np.linspace(-2, 2, 100))),
    "vy": defaultdict(constant_factory(np.linspace(-2, 2, 100))),
    "vz": defaultdict(constant_factory(np.linspace(-20, 20, 100))),
    "beta": defaultdict(constant_factory(np.linspace(0.6, 1.1, 100))),
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

# Every particle residual is an absolute difference (reco - truth).
DEFAULT_PARTICLE_RES_BINS = {
    "beta_res": defaultdict(constant_factory(np.linspace(-0.1, 0.1, 50))),
    "pt_res": defaultdict(constant_factory(np.linspace(-1, 1, 201))),
    "p_res": defaultdict(constant_factory(np.linspace(-1, 1, 201))),
    "eta_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "theta_res": defaultdict(constant_factory(np.linspace(-0.1, 0.1, 50))),
    "phi_res": defaultdict(constant_factory(np.linspace(-0.5, 0.5, 50))),
    "vx_res": defaultdict(constant_factory(np.linspace(-2, 2, 81))),
    "vy_res": defaultdict(constant_factory(np.linspace(-2, 2, 81))),
    # Truth vz is 0 for most matched particles, so this is mostly the reco target
    # profile (flat over roughly -6..0 cm), plus a long tail on the left.
    "vz_res": defaultdict(constant_factory(np.linspace(-15, 5, 81))),
}


# DIS bins configuration
# Placeholder ranges chosen from the ~10.6 GeV fixed-target regime (Q2 up to a few
# GeV^2, 0<xB<1, W up to ~sqrt(s), 0<y<1, nu up to the beam energy). Tune per sample
# via DEFAULT_DIS_BINS[var][Clas12RGASample.CLASDIS_V0] = np.linspace(...).

DEFAULT_DIS_BINS = {
    "Q2": defaultdict(constant_factory(np.linspace(0, 10, 50))),
    "xB": defaultdict(constant_factory(np.linspace(0, 1, 50))),
    "W": defaultdict(constant_factory(np.linspace(1.5, 4.5, 50))),
    "y": defaultdict(constant_factory(np.linspace(0, 1, 50))),
    "nu": defaultdict(constant_factory(np.linspace(0, 10.6, 50))),
}

# The DIS residuals are the only relative ones: (reco - truth) / truth.
DEFAULT_DIS_RES_BINS = {
    "Q2_res": defaultdict(constant_factory(np.linspace(-0.05, 0.05, 50))),
    "xB_res": defaultdict(constant_factory(np.linspace(-0.05, 0.05, 50))),
    # The W/y core is only a few 1e-3 wide (with a long positive tail); keep it resolved.
    "W_res": defaultdict(constant_factory(np.linspace(-0.02, 0.02, 81))),
    "y_res": defaultdict(constant_factory(np.linspace(-0.02, 0.02, 81))),
    "nu_res": defaultdict(constant_factory(np.linspace(-1, 1, 50))),
}


class Clas12RGABinsConfig(BinsConfig[Clas12RGASample]):
    def __init__(
        self,
        event_bins: BinsDictT | None = None,
        event_res_bins: BinsDictT | None = None,
        jet_bins: BinsDictT | None = None,
        jet_res_bins: BinsDictT | None = None,
        particle_bins: BinsDictT | None = None,
        particle_res_bins: BinsDictT | None = None,
        dis_bins: BinsDictT | None = None,
        dis_res_bins: BinsDictT | None = None,
    ):
        super().__init__(
            event_bins=DEFAULT_EVENT_BINS if event_bins is None else event_bins,
            event_res_bins=DEFAULT_EVENT_RES_BINS
            if event_res_bins is None
            else event_res_bins,
            jet_bins=DEFAULT_JET_BINS if jet_bins is None else jet_bins,
            jet_res_bins=DEFAULT_JET_RES_BINS if jet_res_bins is None else jet_res_bins,
            particle_bins=DEFAULT_PARTICLE_BINS
            if particle_bins is None
            else particle_bins,
            particle_res_bins=DEFAULT_PARTICLE_RES_BINS
            if particle_res_bins is None
            else particle_res_bins,
            dis_bins=DEFAULT_DIS_BINS if dis_bins is None else dis_bins,
            dis_res_bins=DEFAULT_DIS_RES_BINS if dis_res_bins is None else dis_res_bins,
        )
