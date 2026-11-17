"""Runtime computation of derived physics variables.

All derived quantities (event sums, species counts, charge, impact parameters,
etc.) are computed at runtime rather than being pre-stored. This ensures
consistency across the file backends.
"""

import numpy as np
import torch
from tqdm import tqdm

# PDG codes for the species-based event variables (Ng, Nn, Pxh, Pyh).
ELECTRON_PDGID = 11
PHOTON_PDGID = 22
NEUTRON_PDGID = 2112
THETA_MIN = 1e-3  # rad; floor for 1 / sin(theta) and tan(theta / 2)

# Per-event species counts derived from pdgId; keys become event-level variable
# suffixes (truth_Ng, pflow_Nn, ...).
SPECIES_COUNTS: dict[str, int] = {"Ng": PHOTON_PDGID, "Nn": NEUTRON_PDGID}


def theta_to_eta(theta: np.ndarray) -> np.ndarray:
    """Polar angle (rad) -> pseudorapidity, ``eta = -ln(tan(theta / 2))``.

    Only for interop with post-eval code that is still written in terms of eta
    (acceptance cut, four-vectors, dR matching, jet clustering). The pipeline
    itself reads ``theta`` straight from the input files, and ``theta`` is what
    gets written back out.

    Args:
        theta: Polar angle in radians. Clamped to ``[THETA_MIN, pi - THETA_MIN]``
            first, since a generated theta is not guaranteed to lie in ``(0, pi)``.

    Returns:
        Pseudorapidity array.
    """
    theta = np.clip(theta, THETA_MIN, np.pi - THETA_MIN)
    return -np.log(np.tan(theta / 2.0))


def segment_sum(
    values: np.ndarray,
    n_particles: np.ndarray,
    cumsum: np.ndarray,
) -> np.ndarray:
    """Per-event sum of a flat particle array, returning 0.0 for empty events.

    ``np.add.reduceat`` cannot be used directly once a collection is allowed to
    have zero particles in an event (the event stage keeps ``npflow == 0``
    events).  For a zero-length segment it returns ``values[start]`` — the *next*
    event's first entry — rather than the additive identity, and it raises
    ``IndexError`` when a trailing segment starts at ``len(values)``.

    Trailing empty events are handled by dropping their start indices before the
    reduction (the last retained segment then correctly runs to the end of
    ``values``); interior empty events are reduced normally and zeroed after.

    Args:
        values: Flat concatenated per-particle array.
        n_particles: Per-event particle counts, shape ``(n_events,)``.
        cumsum: Cumulative sum array, shape ``(n_events + 1,)``.

    Returns:
        Per-event sums, shape ``(n_events,)``, ``float32``.
    """
    n_events = len(n_particles)
    out = np.zeros(n_events, dtype=np.float32)
    if len(values) == 0:
        return out

    starts = cumsum[:-1]
    # Trailing empty events have start == len(values); drop them before reducing.
    n_valid = int(np.searchsorted(starts, len(values), side="left"))
    if n_valid > 0:
        out[:n_valid] = np.add.reduceat(values, starts[:n_valid])
    out[n_particles == 0] = 0.0
    return out


def get_pv_coords(vertex_array: np.ndarray) -> np.ndarray:
    """Get primary vertex coordinates from truth vertices (majority vote).

    Args:
        vertex_array: Array of vertex coordinates from truth particles

    Returns:
        Most common vertex position (primary vertex)
    """
    if len(vertex_array) == 0:
        return np.array(0.0)  # Default to zero if no vertices are present
    values, counts = np.unique(vertex_array, return_counts=True)
    pv_index = np.argmax(counts)
    pv_coords = values[pv_index]
    return pv_coords


def calculate_impact_params(
    eta: np.ndarray,
    phi: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
    v_z: np.ndarray,
    pv_x: float | np.ndarray,
    pv_y: float | np.ndarray,
    pv_z: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate d0 and z0 impact parameters from particle kinematics and vertices.

    Args:
        eta: Particle pseudorapidity
        phi: Particle azimuthal angle
        v_x, v_y, v_z: Particle vertex coordinates
        pv_x, pv_y, pv_z: Primary vertex coordinates

    Returns:
        Tuple of (d0, z0) impact parameter arrays
    """
    # 1. Coordinate shift (Particle Vertex - Primary Vertex)
    delta_x = v_x - pv_x
    delta_y = v_y - pv_y
    delta_z = v_z - pv_z

    # 2. Momentum components
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # 3. Calculate d0 (transverse impact parameter)
    d0 = -delta_x * sin_phi + delta_y * cos_phi

    # 4. Calculate z0 (longitudinal impact parameter)
    # L_xy = projection of delta_r onto transverse momentum direction
    L_xy = delta_x * cos_phi + delta_y * sin_phi

    # z0 is delta_z adjusted by the slope (sinh(eta)) over the distance L_xy
    z0 = delta_z - L_xy * np.sinh(eta)

    return d0, z0


class ComputedVariables:
    """Computes all derived physics variables at runtime.

    All methods are stateless and operate on numpy arrays.
    compute_flat_* methods expect pre-flattened concatenated arrays (fast, vectorized).
    compute_ragged_* methods expect ragged per-event object arrays.

    Supports:
    - d0Sig, z0Sig: Impact parameter significance (IP / error)
    - means: Per-event mean of context variables
    - ht_p, charge: Per-event scalar momentum sum; per-particle charge
    - Ng, Nn: Per-event photon / neutron multiplicity, from pdgId
    - Pxh, Pyh: Per-event hadronic (non-electron) px / py sum, from pdgId
    - PV coordinates, vz_shifted, d0My/z0My: From vertex positions
    """

    @staticmethod
    def compute_significance(
        d0: np.ndarray | torch.Tensor,
        d0_error: np.ndarray | torch.Tensor,
        z0: np.ndarray | torch.Tensor,
        z0_error: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Compute impact parameter significance, clipped to ±500.

        Args:
            d0, z0: Impact parameter arrays
            d0_error, z0_error: Impact parameter uncertainties

        Returns:
            Tuple of (d0Sig, z0Sig) arrays
        """
        d0_sig = d0 / d0_error
        z0_sig = z0 / z0_error
        if isinstance(d0_sig, np.ndarray) and isinstance(z0_sig, np.ndarray):
            d0_sig = np.clip(d0_sig, -500, 500)
            z0_sig = np.clip(z0_sig, -500, 500)
        elif isinstance(d0_sig, torch.Tensor) and isinstance(z0_sig, torch.Tensor):
            d0_sig = torch.clamp(d0_sig, -500, 500)
            z0_sig = torch.clamp(z0_sig, -500, 500)
        return d0_sig, z0_sig

    @staticmethod
    def compute_flat_means(
        particle_data: dict[str, np.ndarray],
        collection_name: str,
        var_names: list[str],
        n_particles: np.ndarray,
        cumsum: np.ndarray,
    ) -> np.ndarray:
        """Compute per-event means from flat concatenated arrays.

        Used by the eager (ROOT) dataset at init time for pre-computing means
        across the entire dataset using efficient ``np.add.reduceat`` operations.

        For per-batch tensor-based means (shape-agnostic, works on any batch
        dimension), use :func:`parnassus_core.data.physics_data.compute_means`.

        Excludes variables containing 'class' or 'charge' from the mean calculation.

        Args:
            particle_data: Dictionary of flat (concatenated) particle arrays
            collection_name: Collection to compute means for (e.g., 'truth')
            var_names: List of variable names (with collection prefix)
            n_particles: Per-event particle counts, shape ``(n_events,)``
            cumsum: Cumulative sum array, shape ``(n_events + 1,)``

        Returns:
            Array of shape (n_events, n_vars) with per-event means
        """
        # Filter out class, charge, and pdgId variables
        filtered_vars = [
            v
            for v in var_names
            if v.startswith(f"{collection_name}_")
            and "class" not in v
            and "charge" not in v
            and "pdgId" not in v
        ]

        n_events = len(n_particles)
        n_vars = len(filtered_vars)
        means = np.zeros((n_events, n_vars), dtype=np.float32)

        for i, var in enumerate(tqdm(filtered_vars, desc="Calculating means")):
            # segment_sum tolerates empty events; the mean of one is left at 0.
            means[:, i] = np.divide(
                segment_sum(particle_data[var], n_particles, cumsum),
                n_particles,
                out=np.zeros(n_events, dtype=np.float32),
                where=n_particles > 0,
            )

        return means

    @staticmethod
    def compute_flat_derived(
        p_flat: np.ndarray,
        theta_flat: np.ndarray,
        phi_flat: np.ndarray,
        n_particles: np.ndarray,
        cumsum: np.ndarray,
        pdgId_flat: np.ndarray | None = None,
        class_flat: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute the derived event variables and charge from flat arrays.

        Uses segment sums for per-event aggregation — no Python per-event loop.

        Args:
            p_flat: Flat concatenated momentum magnitude for all events.
            theta_flat: Flat concatenated polar angle (rad).
            phi_flat: Flat concatenated azimuthal angle (rad).
            n_particles: Per-event particle counts, shape ``(n_events,)``.
            cumsum: Cumulative sum array, shape ``(n_events + 1,)``.
            pdgId_flat: Flat concatenated pdgId (optional, for charge and the
                species-based event variables).
            class_flat: Flat concatenated class (optional, for charge).

        Returns:
            Dict with the event-level ``ht_p`` (scalar sum of p, shape
            ``(n_events,)``) and optionally the per-particle ``charge``. When
            ``pdgId_flat`` is supplied, the event-level species counts (``Ng``,
            ``Nn`` — see ``SPECIES_COUNTS``) and hadronic momentum sums
            (``Pxh``, ``Pyh``) are added.
        """
        # segment_sum yields 0.0 for empty events.
        result: dict[str, np.ndarray] = {
            "ht_p": segment_sum(p_flat, n_particles, cumsum),
        }

        if pdgId_flat is not None:
            result.update(
                ComputedVariables.compute_flat_counts(pdgId_flat, n_particles, cumsum)
            )
            result.update(
                ComputedVariables.compute_flat_hadronic_missing_p(
                    pdgId_flat, p_flat, theta_flat, phi_flat, n_particles, cumsum
                )
            )

        if pdgId_flat is not None and class_flat is not None:
            charge = np.sign(pdgId_flat).astype(np.float32)
            # charge[class_flat > 2] = 0
            result["charge"] = charge

        return result

    @staticmethod
    def compute_flat_counts(
        pdgid_flat: np.ndarray,
        n_particles: np.ndarray,
        cumsum: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Per-event multiplicity of each species in ``SPECIES_COUNTS``.

        Species are matched on the exact pdgId, so antiparticles are not folded
        into their particle's count.

        Args:
            pdgid_flat: Flat concatenated pdgId array.
            n_particles: Per-event particle counts, shape ``(n_events,)``.
            cumsum: Cumulative sum array, shape ``(n_events + 1,)``.

        Returns:
            Dict keyed by ``SPECIES_COUNTS`` name (``Ng``, ``Nn``), each shape
            ``(n_events,)``, ``float32``.  Empty events count 0.
        """
        return {
            name: segment_sum(
                (pdgid_flat == pdgid).astype(np.float32), n_particles, cumsum
            )
            for name, pdgid in SPECIES_COUNTS.items()
        }

    @staticmethod
    def compute_flat_hadronic_missing_p(
        pdgid_flat: np.ndarray,
        p_flat: np.ndarray,
        theta_flat: np.ndarray,
        phi_flat: np.ndarray,
        n_particles: np.ndarray,
        cumsum: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Per-event transverse momentum sum of the hadronic system.

        The vector sum of ``px = p sin(theta) cos(phi)`` and
        ``py = p sin(theta) sin(phi)`` over every particle that is not an
        electron (``pdgId == 11``); positrons are kept.  This is the visible
        hadronic momentum, not its negative (the "missing" momentum).

        Args:
            pdgid_flat: Flat concatenated pdgId array.
            p_flat: Flat concatenated momentum magnitude.
            theta_flat: Flat concatenated polar angle (rad).
            phi_flat: Flat concatenated azimuthal angle (rad).
            n_particles: Per-event particle counts, shape ``(n_events,)``.
            cumsum: Cumulative sum array, shape ``(n_events + 1,)``.

        Returns:
            Dict with keys ``Pxh`` and ``Pyh``, each shape ``(n_events,)``,
            ``float32``.  Events with no hadrons get ``Pxh = 0`` and
            ``Pyh = 0``.
        """
        # Zero electrons rather than dropping them, so the arrays stay aligned
        # with the per-event cumsum offsets.
        is_hadron = pdgid_flat != ELECTRON_PDGID
        sin_theta = np.sin(theta_flat)
        px = np.where(is_hadron, p_flat * sin_theta * np.cos(phi_flat), 0.0)
        py = np.where(is_hadron, p_flat * sin_theta * np.sin(phi_flat), 0.0)
        return {
            "Pxh": segment_sum(px, n_particles, cumsum),
            "Pyh": segment_sum(py, n_particles, cumsum),
        }

    @staticmethod
    def compute_ragged_pv_coords(
        truth_vx: np.ndarray,
        truth_vy: np.ndarray,
        truth_vz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute primary vertex coordinates from ragged truth vertex arrays.

        PV is determined per event by majority vote on truth vertex positions.
        Only truth particles carry the true vertex info.

        Args:
            truth_vx, truth_vy, truth_vz: Ragged arrays of truth vertex
                coordinates, each of shape ``(nevents,)`` with sub-arrays.

        Returns:
            ``(pv_x, pv_y, pv_z)`` — each ``np.float32`` array of shape
            ``(nevents,)``.
        """
        nevents = len(truth_vx)
        pv_x = np.empty(nevents, dtype=np.float32)
        pv_y = np.empty(nevents, dtype=np.float32)
        pv_z = np.empty(nevents, dtype=np.float32)

        for i in tqdm(range(nevents), desc="Computing PV coordinates"):
            pv_x[i] = get_pv_coords(truth_vx[i])
            pv_y[i] = get_pv_coords(truth_vy[i])
            pv_z[i] = get_pv_coords(truth_vz[i])

        return pv_x, pv_y, pv_z

    @staticmethod
    def compute_ragged_vz_shifted(
        vz_ragged: np.ndarray,
        pv_z: np.ndarray,
    ) -> list[np.ndarray]:
        """Shift vertex z by the primary vertex z, for ragged per-event arrays.

        Args:
            vz_ragged: Ragged array of vertex z, shape ``(nevents,)``.
            pv_z: PV z-coordinate per event, shape ``(nevents,)``.

        Returns:
            List of per-event ``vz - pv_z`` arrays.
        """
        return [vz_ragged[i] - pv_z[i] for i in range(len(pv_z))]

    @staticmethod
    def compute_ragged_ip_params(
        eta_ragged: np.ndarray,
        phi_ragged: np.ndarray,
        vx_ragged: np.ndarray,
        vy_ragged: np.ndarray,
        vz_ragged: np.ndarray,
        pv_x: np.ndarray,
        pv_y: np.ndarray,
        pv_z: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute d0My and z0My impact parameters from ragged per-event arrays.

        Only meaningful for reconstructed (pflow) particles, not truth.

        Args:
            eta_ragged, phi_ragged: Ragged kinematic arrays.
            vx_ragged, vy_ragged, vz_ragged: Ragged vertex arrays.
            pv_x, pv_y, pv_z: PV coordinates, shape ``(nevents,)``.

        Returns:
            ``(d0My_list, z0My_list)`` — lists of per-event arrays.
        """
        d0_list: list[np.ndarray] = []
        z0_list: list[np.ndarray] = []

        for i in tqdm(range(len(pv_x)), desc="Computing impact parameters"):
            d0, z0 = calculate_impact_params(
                eta_ragged[i],
                phi_ragged[i],
                vx_ragged[i],
                vy_ragged[i],
                vz_ragged[i],
                pv_x[i],
                pv_y[i],
                pv_z[i],
            )
            d0_list.append(d0)
            z0_list.append(z0)

        return d0_list, z0_list
