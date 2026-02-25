"""Runtime computation of derived physics variables.

All derived quantities (ptrel, charge, impact parameters, etc.) are computed
at runtime rather than being pre-stored. This ensures consistency across
ROOT and HDF5 backends.
"""

import numpy as np
import torch
from tqdm import tqdm


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

    This class centralizes computation logic that was previously duplicated
    between ROOT and HDF5 backends. All methods are stateless and operate
    on numpy arrays.

    Supports:
    - ptrel: Relative transverse momentum (pt / ht)
    - charge: Particle charge from pdgId
    - d0My, z0My: Recalculated impact parameters from vertex positions
    - d0Sig, z0Sig: Impact parameter significance (IP / error)
    - vz_shifted: Vertex z-coordinate shifted by primary vertex
    - means: Per-event mean of context variables
    """

    @staticmethod
    def compute_ptrel(pt: np.ndarray, ht: float | np.ndarray) -> np.ndarray:
        """Compute relative transverse momentum (pt / ht).

        Args:
            pt: Particle transverse momentum array
            ht: Scalar sum of all transverse momenta in event

        Returns:
            Relative pt array (pt/ht), or zeros if ht=0
        """
        if isinstance(ht, np.ndarray):
            # Handle batch case
            return np.divide(pt, ht, out=np.zeros_like(pt), where=(ht > 0))
        return pt / ht if ht > 0 else pt * 0

    @staticmethod
    def compute_charge(pdgId: np.ndarray, class_: np.ndarray) -> np.ndarray:
        """Compute particle charge from PDG ID.

        Args:
            pdgId: PDG particle ID array
            class_: Particle class (0-4, where >2 are neutral)

        Returns:
            Charge array (+1, 0, -1), with neutrals forced to 0

        Note:
            If charge is pre-stored in the file, use that instead.
        """
        charge = np.sign(pdgId)
        charge[class_ > 2] = 0  # Force neutrals to zero charge
        return charge

    @staticmethod
    def compute_impact_params(
        eta: np.ndarray,
        phi: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        pv_x: float | np.ndarray,
        pv_y: float | np.ndarray,
        pv_z: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute d0My and z0My from vertex positions and primary vertex.

        These share scalers with d0/z0 (the "My" suffix is removed in scaler_key).

        Args:
            eta, phi: Particle kinematics
            vx, vy, vz: Particle vertex coordinates
            pv_x, pv_y, pv_z: Primary vertex coordinates

        Returns:
            Tuple of (d0My, z0My) arrays
        """
        return calculate_impact_params(eta, phi, vx, vy, vz, pv_x, pv_y, pv_z)

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
    def compute_vz_shifted(vz: np.ndarray, pv_z: float | np.ndarray) -> np.ndarray:
        """Shift vertex z-coordinate by primary vertex position.

        Args:
            vz: Particle vertex z-coordinate
            pv_z: Primary vertex z-coordinate

        Returns:
            Shifted z-coordinate (vz - pv_z)
        """
        return vz - pv_z

    @staticmethod
    def compute_means_bulk(
        particle_data: dict[str, np.ndarray],
        collection_name: str,
        var_names: list[str],
        n_particles: np.ndarray,
        cumsum: np.ndarray,
    ) -> np.ndarray:
        """Compute per-event means over all events at once (bulk, numpy-based).

        Used by the eager (ROOT) dataset at init time for pre-computing means
        across the entire dataset using efficient ``np.add.reduceat`` operations.

        For per-batch tensor-based means (shape-agnostic, works on any batch
        dimension), use :func:`parnassus_core.data.physics_data.compute_means`.

        Excludes variables containing 'class' or 'charge' from the mean calculation.

        Args:
            particle_data: Dictionary of concatenated particle arrays
            collection_name: Collection to compute means for (e.g., 'truth')
            var_names: List of variable names (with collection prefix)
            n_particles: Array of particle counts per event
            cumsum: Cumulative sum for indexing

        Returns:
            Array of shape (n_events, n_vars) with per-event means
        """
        # Filter out class and charge variables
        filtered_vars = [
            v
            for v in var_names
            if v.startswith(f"{collection_name}_")
            and "class" not in v
            and "charge" not in v
        ]

        n_events = len(n_particles)
        n_vars = len(filtered_vars)
        means = np.zeros((n_events, n_vars), dtype=np.float32)

        for i, var in enumerate(tqdm(filtered_vars, desc="Calculating means")):
            # Use reduceat to efficiently compute sums for each event
            means[:, i] = (
                np.add.reduceat(
                    particle_data[var],
                    cumsum[:-1],
                )
                / n_particles
            )

        return means

    @staticmethod
    def compute_event_aggregates(
        pt: np.ndarray, phi: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute event-level aggregates (ht, met_x, met_y).

        Args:
            pt: Array of particle transverse momenta
            phi: Array of particle azimuthal angles

        Returns:
            Tuple of (ht, met_x, met_y)
        """
        met_x = (pt * np.cos(phi)).sum()
        met_y = (pt * np.sin(phi)).sum()
        ht = pt.sum()
        return ht, met_x, met_y

    @staticmethod
    def compute_bulk_derived(
        pt_flat: np.ndarray,
        phi_flat: np.ndarray,
        n_particles: np.ndarray,
        cumsum: np.ndarray,
        pdgId_flat: np.ndarray | None = None,
        class_flat: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute ht, met_x, met_y, ptrel, charge over all events (vectorized).

        Uses ``np.add.reduceat`` for efficient per-event aggregation over
        the flat concatenated particle arrays — no Python per-event loop.

        Args:
            pt_flat: Flat concatenated pt array for all events.
            phi_flat: Flat concatenated phi array.
            n_particles: Per-event particle counts, shape ``(n_events,)``.
            cumsum: Cumulative sum array, shape ``(n_events + 1,)``.
            pdgId_flat: Flat concatenated pdgId (optional, for charge).
            class_flat: Flat concatenated class (optional, for charge).

        Returns:
            Dict with keys ``ht``, ``met_x``, ``met_y`` (event-level,
            shape ``(n_events,)``), ``ptrel`` (particle-level, same
            length as ``pt_flat``), and optionally ``charge``.
        """
        n_events = len(n_particles)

        # Event-level aggregates via reduceat
        ht = np.add.reduceat(pt_flat, cumsum[:-1]).astype(np.float32)
        met_x = np.add.reduceat(pt_flat * np.cos(phi_flat), cumsum[:-1]).astype(
            np.float32
        )
        met_y = np.add.reduceat(pt_flat * np.sin(phi_flat), cumsum[:-1]).astype(
            np.float32
        )

        # ptrel: broadcast event-level ht to particle level
        event_idx = np.repeat(np.arange(n_events), n_particles)
        ht_per_particle = ht[event_idx]
        ptrel = np.divide(
            pt_flat,
            ht_per_particle,
            out=np.zeros_like(pt_flat, dtype=np.float32),
            where=ht_per_particle > 0,
        )

        result: dict[str, np.ndarray] = {
            "ht": ht,
            "met_x": met_x,
            "met_y": met_y,
            "ptrel": ptrel,
        }

        if pdgId_flat is not None and class_flat is not None:
            charge = np.sign(pdgId_flat).astype(np.float32)
            charge[class_flat > 2] = 0
            result["charge"] = charge

        return result

    @staticmethod
    def compute_bulk_pv_coords(
        truth_vx: np.ndarray,
        truth_vy: np.ndarray,
        truth_vz: np.ndarray,
        nevents: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute primary vertex coordinates from truth particles.

        PV is determined per event by majority vote on truth vertex
        positions.  Only truth particles carry the true vertex info.

        Args:
            truth_vx, truth_vy, truth_vz: Ragged arrays of truth vertex
                coordinates, each of shape ``(nevents,)`` with sub-arrays.
            nevents: Number of events.

        Returns:
            ``(pv_x, pv_y, pv_z)`` — each ``np.float32`` array of shape
            ``(nevents,)``.
        """
        pv_x = np.empty(nevents, dtype=np.float32)
        pv_y = np.empty(nevents, dtype=np.float32)
        pv_z = np.empty(nevents, dtype=np.float32)

        for i in tqdm(range(nevents), desc="Computing PV coordinates"):
            pv_x[i] = get_pv_coords(truth_vx[i])
            pv_y[i] = get_pv_coords(truth_vy[i])
            pv_z[i] = get_pv_coords(truth_vz[i])

        return pv_x, pv_y, pv_z

    @staticmethod
    def compute_bulk_vz_shifted(
        vz_ragged: np.ndarray,
        pv_z: np.ndarray,
        nevents: int,
    ) -> list[np.ndarray]:
        """Compute vz_shifted for any collection (truth or pflow).

        Simply shifts each particle's vertex z by the event's PV z.

        Args:
            vz_ragged: Ragged array of vertex z, shape ``(nevents,)``.
            pv_z: PV z-coordinate per event, shape ``(nevents,)``.
            nevents: Number of events.

        Returns:
            List of per-event shifted arrays.
        """
        return [vz_ragged[i] - pv_z[i] for i in range(nevents)]

    @staticmethod
    def compute_bulk_ip_params(
        eta_ragged: np.ndarray,
        phi_ragged: np.ndarray,
        vx_ragged: np.ndarray,
        vy_ragged: np.ndarray,
        vz_ragged: np.ndarray,
        pv_x: np.ndarray,
        pv_y: np.ndarray,
        pv_z: np.ndarray,
        nevents: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute d0My and z0My impact parameters (pflow-only).

        Impact parameters quantify the distance of closest approach of
        a reconstructed track to the primary vertex.  Only meaningful
        for reconstructed (pflow) particles, not truth.

        Args:
            eta_ragged, phi_ragged: Ragged kinematic arrays.
            vx_ragged, vy_ragged, vz_ragged: Ragged vertex arrays.
            pv_x, pv_y, pv_z: PV coordinates, shape ``(nevents,)``.
            nevents: Number of events.

        Returns:
            ``(d0My_list, z0My_list)`` — lists of per-event arrays.
        """
        d0_list: list[np.ndarray] = []
        z0_list: list[np.ndarray] = []

        for i in tqdm(range(nevents), desc="Computing impact parameters"):
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
