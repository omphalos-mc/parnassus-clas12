import numpy as np
import pytest

from parnassus_core.data.utils.computed_vars import ComputedVariables

# Two events: (e-, pi+, gamma) and (n); theta/phi in rad.
P = np.array([4.0, 1.0, 0.5, 2.0])
THETA = np.array([0.2, 0.5, 1.0, 0.3])
PHI = np.array([1.0, -2.0, 0.5, 3.0])
PDG_ID = np.array([11, 211, 22, 2112])
N_PARTICLES = np.array([3, 1])
CUMSUM = np.array([0, 3, 4])


def test_p_based_event_variables():
    derived = ComputedVariables.compute_flat_derived(
        P, THETA, PHI, N_PARTICLES, CUMSUM, pdgId_flat=PDG_ID
    )
    px = P * np.sin(THETA) * np.cos(PHI)
    py = P * np.sin(THETA) * np.sin(PHI)
    assert derived["ht_p"] == pytest.approx([5.5, 2.0])
    # The electron is excluded from the hadronic sums.
    assert derived["Pxh"] == pytest.approx([px[1] + px[2], px[3]], rel=1e-6)
    assert derived["Pyh"] == pytest.approx([py[1] + py[2], py[3]], rel=1e-6)
    assert derived["Ng"].tolist() == [1, 0]
    assert derived["Nn"].tolist() == [0, 1]
    assert not {"ht", "met_x", "met_y", "ptrel"} & derived.keys()
