"""Particle-ID (pdgId) mappings

Dependency-light home for the PDG-code physics tables shared across the data
pipeline (``data/utils/transforms.py``, ``data/utils/scaling.py``) and the
preprocessing converters. Intentionally free of heavy / optional imports
(no ``energyflow`` / ``uproot`` / ``vector``); ``torch`` is imported lazily
inside the encode/decode helpers, so importing this module never requires torch.

Provides:

- ``pdgid_class_dict`` -- pdgId -> coarse particle class (0-6, -1 for neutrinos).
- ``pid_onehot_ids`` / ``N_PID_CLASSES`` and ``pdgid_to_onehot_index`` /
  ``onehot_index_to_pdgid`` -- pdgId <-> contiguous one-hot index used for the
  CLAS12 PID prediction target.
"""

### see https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
###  6: quarks
### -1: neutrinos

pdgid_class_dict = {
    0: 5,
    -211: 0,
    211: 0,  # pi+-
    -321: 0,
    321: 0,  # kaon+-
    -411: 0,
    411: 0,  # D+-
    -431: 0,
    431: 0,  # D_s+-
    -521: 0,
    521: 0,  # B+-
    -2212: 0,
    2212: 0,  # proton
    -3112: 0,
    3112: 0,  # sigma-
    -3312: 0,
    3312: 0,  # xi+-
    -3222: 0,
    3222: 0,  # sigma+
    -3334: 0,
    3334: 0,  # omega
    -11: 1,
    11: 1,  # e
    -13: 2,
    13: 2,  # mu
    -111: 3,
    111: 3,  # pi0
    130: 3,  # K0L
    310: 3,  # K0S
    -2112: 3,
    2112: 3,  # neutrons
    -3122: 3,
    3122: 3,  # lambda
    -3322: 3,
    3322: 3,  # xi0
    22: 4,  # photon
    45: 0,  # deuteron (CLAS12 shortened id; see pid_onehot_ids)
    1000010020: 0,  # deuteron (PDG nuclear code; used by CMS preprocessing)
    1000010030: 0,  # triton
    1000010040: 0,  # alpha
    1000020030: 0,  # He3
    1000020040: 0,  # He4
    1000030040: 0,  # Li6
    1000030050: 0,  # Li7
    -999: 5,  # residual
    -12: -1,
    12: -1,  # nu_e
    -14: -1,
    14: -1,  # nu_mu
    -16: -1,
    16: -1,  # nu_tau
    -2: 6,
    2: 6,  # u
    -4: 6,
    4: 6,  # c
    -1: 6,
    1: 6,  # d
    -3: 6,
    3: 6,  # s
    -5: 6,
    5: 6,  # b
    -6: 6,
    6: 6,  # t
}


# ---------------------------------------------------------------------------
# pdgId <-> one-hot index mapping (CLAS12_RGA PID prediction)
# ---------------------------------------------------------------------------
# Ordered list of pdgIds that each get a dedicated one-hot bucket; the position
# in this list IS the one-hot index. Any pdgId not listed maps to the trailing
# "other" bucket at index ``_OTHER_PID_INDEX``. Covers the reconstructed pflow
# species observed in the v1 CLAS12_RGA sample. Note: the deuteron uses the
# shortened CLAS12 id ``45`` (not the PDG nuclear code 1000010020) so every id
# here is exactly representable in ``float32``.
pid_onehot_ids = [22, 11, -11, 211, -211, 321, -321, 2212, -2212, 2112, 45]
N_PID_CLASSES = len(pid_onehot_ids) + 1  # +1 for the "other" bucket -> 12
_OTHER_PID_INDEX = len(pid_onehot_ids)  # index of the "other" bucket -> 11


def pdgid_to_onehot_index(pdgid):
    """Map pdgIds to contiguous one-hot indices.

    Unknown pdgIds map to the trailing "other" bucket (``_OTHER_PID_INDEX``).

    Accepts an integer or ``float32`` tensor (the data pipeline casts every
    feature with ``.float()``); values are cast to ``long`` for comparison. All
    supported pdgIds are small enough to be exact in ``float32`` (the deuteron
    uses the shortened CLAS12 id ``45``), so the ``long`` cast is lossless.

    Parameters
    ----------
    pdgid : torch.Tensor
        pdgId values (any shape), integer or float dtype.

    Returns
    -------
    torch.Tensor
        One-hot indices in ``[0, N_PID_CLASSES)``, same shape as ``pdgid``.
    """
    import torch

    pdgid_long = pdgid.long()
    idx = torch.full_like(pdgid_long, _OTHER_PID_INDEX)
    for i, pid in enumerate(pid_onehot_ids):
        idx[pdgid_long == pid] = i
    return idx


def onehot_index_to_pdgid(index):
    """Inverse of :func:`pdgid_to_onehot_index`.

    Maps one-hot indices back to pdgId values; the "other" bucket maps to 0.

    Parameters
    ----------
    index : torch.Tensor
        One-hot indices in ``[0, N_PID_CLASSES)`` (any shape).

    Returns
    -------
    torch.Tensor
        pdgId values (``long``), same shape as ``index`` ("other" -> 0).
    """
    import torch

    lut = torch.tensor([*pid_onehot_ids, 0], dtype=torch.long, device=index.device)
    return lut[index.long()]


def pdgid_to_class(pdgid, default=5):
    """Map pdgId values to coarse particle-class ids via ``pdgid_class_dict``.

    Used by the inverse-scaling path (``data/utils/scaling.py``) to recover the
    class id needed for class-based kinematic unscaling when the categorical
    target is ``pdgId`` rather than ``class`` (the CLAS12 PID config). Unknown
    pdgIds map to ``default`` (5, "residual"); with the standard
    ``pid_onehot_ids`` every decoded value (including the deuteron id ``45`` and
    the "other" bucket's ``0``) is present in ``pdgid_class_dict``, so ``default``
    is only a safety fallback.

    Parameters
    ----------
    pdgid : torch.Tensor
        pdgId values (any shape), integer or float dtype.
    default : int, optional
        Class id for pdgIds absent from ``pdgid_class_dict`` (default 5).

    Returns
    -------
    torch.Tensor
        Class ids (``long``), same shape as ``pdgid``.
    """
    import torch

    pdgid_long = pdgid.long()
    out = torch.full_like(pdgid_long, default)
    for pid, cls in pdgid_class_dict.items():
        out[pdgid_long == pid] = cls
    return out


# ---------------------------------------------------------------------------
# pdgId -> rest mass (GeV)
# ---------------------------------------------------------------------------
# Rest masses in GeV for the reconstructed pflow species (see ``pid_onehot_ids``).
# Used by the postprocessing energy refinement (E = sqrt(p^2 + m^2)) and DIS
# reconstruction; everything not listed falls back to ``default`` (0.0, massless).
# The deuteron uses the shortened CLAS12 id ``45``.
pdgid_mass_dict = {
    11: 0.00051099891,
    -11: 0.00051099891,  # e
    13: 0.10566,
    -13: 0.10566,  # mu
    211: 0.13957000,
    -211: 0.13957000,  # pi+-
    321: 0.49367700,
    -321: 0.49367700,  # kaon+-
    2212: 0.93827200,
    -2212: 0.93827200,  # proton
    2112: 0.93956500,
    -2112: 0.93956500,  # neutron
    22: 0.0,  # photon
    45: 1.875612,  # deuteron (CLAS12 shortened id)
}


def pdgid_to_mass(pdgid, default=0.0):
    """Map pdgId values to rest mass in GeV via ``pdgid_mass_dict``.

    Torch-free numpy helper (mirrors :func:`pdgid_to_class`) used by the
    postprocessing energy refinement and DIS reconstruction. Unknown pdgIds map
    to ``default`` (0.0, i.e. treated as massless).

    Parameters
    ----------
    pdgid : int or numpy.ndarray
        pdgId value(s), integer or float dtype.
    default : float, optional
        Mass (GeV) for pdgIds absent from ``pdgid_mass_dict`` (default 0.0).

    Returns
    -------
    float or numpy.ndarray
        Rest mass(es) in GeV, same shape as ``pdgid``.
    """
    import numpy as np

    pdgid_arr = np.asarray(pdgid)
    pdgid_int = np.rint(pdgid_arr).astype(np.int64)
    out = np.full(pdgid_int.shape, default, dtype=np.float64)
    for pid, mass in pdgid_mass_dict.items():
        out[pdgid_int == pid] = mass
    return out if out.ndim else out.item()
