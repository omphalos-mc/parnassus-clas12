"""Shared processing logic for physics datasets.

Both eager (ROOT) and streaming (HDF5) dataset implementations call these
functions after preparing data in the **common intermediate format**:

Common Intermediate Format
--------------------------
``collection_data : dict[str, dict[str, Tensor]]``
    ``{collection_name: {bare_var_name: Tensor(…, N)}}``
    ROOT single-event: tensors of shape ``(N,)`` where ``N = max_particles``
    HDF5 batch: tensors of shape ``(B, N)``

``collection_masks : dict[str, Tensor]``
    ``{collection_name: BoolTensor(…, N)}``
    True for valid (real) particles, False for padding.

``event_data : dict[str, Tensor]``
    ``{var_name: Tensor(…,)}``
    ROOT single-event: scalar tensor ``()``
    HDF5 batch: ``(B,)``

Each dataset class is responsible for:

1. Loading raw data from its backend
2. Extracting, sorting, and padding (ROOT) or reading pre-padded (HDF5)
3. Packaging into the common intermediate format above
4. Calling shared functions below for transform + assembly

These shared functions handle:

- Variable routing (collection prefix → data lookup)
- Transform application (sin/cos, one-hot, scaling)
- Feature concatenation
- Backward-compatible output key naming
"""

from typing import Literal

import torch
from torch import Tensor

from parnassus_core.data.utils.transforms import apply_variable_transform
from parnassus_core.data.utils.variable_helper import VariableHelper
from parnassus_core.data.utils.scaling import FeatureScaler


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def do_padding(tensor: Tensor, max_len: int) -> Tensor:
    """Pad tensor to *max_len* along the first dimension.

    Used by the eager (ROOT) dataset to pad variable-length particle arrays
    to a fixed size.  Streaming (HDF5) datasets receive pre-padded data and
    do not call this function.

    Args:
        tensor: Input tensor of shape ``(n, ...)`` where ``n <= max_len``.
        max_len: Maximum length to pad to.

    Returns:
        Padded tensor of shape ``(max_len, ...)``.
    """
    n = tensor.shape[0]
    if n >= max_len:
        return tensor[:max_len]
    pad_shape = (max_len - n,) + tensor.shape[1:]
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=0)


# ---------------------------------------------------------------------------
# Build functions — shared by both eager and streaming datasets
# ---------------------------------------------------------------------------


def build_ctxt_data(
    ctxt_vars: list[str],
    collection_data: dict[str, dict[str, Tensor]],
    collection_masks: dict[str, Tensor],
    sin_cos: bool,
    feature_scaler: FeatureScaler,
    var_helper: VariableHelper,
    ctxt_mask_type: Literal["all", "charged", "neutral"] = "all",
) -> dict[str, Tensor]:
    """Build context data from particle collections.

    Groups variables by collection prefix, applies transforms (sin/cos,
    one-hot, scaling) to each, and concatenates features.  Collections
    not present in *collection_data* are silently skipped (e.g. pflow
    in eval mode).

    Args:
        ctxt_vars: Variable names with collection prefixes
            (e.g. ``["truth_ptrel", "truth_eta", "pflow_ptrel"]``).
        collection_data: ``{collection: {bare_var: Tensor(…, N)}}``.
        collection_masks: ``{collection: BoolTensor(…, N)}``.
        sin_cos: Apply sin/cos transform to phi variables.
        feature_scaler: :class:`FeatureScaler` instance.
        var_helper: :class:`VariableHelper` with registered collections.
        ctxt_mask_type: ``"all"``, ``"charged"``, or ``"neutral"`` to filter
            context particles by class (requires "class" variable in collection).

    Returns:
        Dict with backward-compatible keys:

        - 1 collection  → ``{ctxt_data, ctxt_mask}``
        - 2 collections → ``{ctxt_data, ctxt_mask, pf_ctxt_data, pf_ctxt_mask}``

        Shapes — ROOT: ``(N, F)``; HDF5: ``(B, N, F)``.
    """
    # Group variables by collection, preserving insertion order
    vars_by_coll: dict[str, list[str]] = {}
    for var in ctxt_vars:
        coll = var_helper.get_collection(var)
        if coll is None:
            raise ValueError(
                f"Context variable '{var}' must have a collection prefix "
                f"(one of: {', '.join(var_helper.collection_names)})"
            )
        vars_by_coll.setdefault(coll, []).append(var)

    # Build features for each collection present in collection_data
    result_data: dict[str, Tensor] = {}
    result_masks: dict[str, Tensor] = {}
    coll_order: list[str] = []

    for coll_name, var_list in vars_by_coll.items():
        if coll_name not in collection_data:
            continue  # Skip missing collections (e.g. pflow in eval mode)

        coll_order.append(coll_name)
        data = collection_data[coll_name]
        mask = collection_masks[coll_name]

        features: list[Tensor] = []
        for var in var_list:
            base = var_helper.get_base_name(var)
            x = data[base].float().unsqueeze(-1)  # (…, N) → (…, N, 1)
            transformed = apply_variable_transform(
                var, x, sin_cos, feature_scaler, var_helper
            )
            features.append(transformed)

        # Apply mask-type filtering for charged/neutral subsets
        if ctxt_mask_type != "all" and "class" in data:
            class_vals = data["class"]  # (…, N)
            if ctxt_mask_type == "charged":
                mask = mask & (class_vals < 3)
            elif ctxt_mask_type == "neutral":
                mask = mask & (class_vals >= 3)

        feat_tensor = torch.cat(features, dim=-1)  # (…, N, F)
        # Zero out padding: legacy code applies transforms BEFORE padding,
        # so padded positions must be exactly zero after transforms.
        feat_tensor = feat_tensor * mask.unsqueeze(-1).float()
        result_data[coll_name] = feat_tensor.nan_to_num(0.0)
        result_masks[coll_name] = mask.bool()

    # Backward-compatible key naming
    # First collection  → ctxt_data / ctxt_mask
    # Second collection → pf_ctxt_data / pf_ctxt_mask
    result: dict[str, Tensor] = {}
    if len(coll_order) >= 1:
        result["ctxt_data"] = result_data[coll_order[0]]
        result["ctxt_mask"] = result_masks[coll_order[0]]
    if len(coll_order) >= 2:
        result["pf_ctxt_data"] = result_data[coll_order[1]]
        result["pf_ctxt_mask"] = result_masks[coll_order[1]]

    return result


def build_fs_data(
    fs_vars: list[str],
    collection_data: dict[str, dict[str, Tensor]],
    collection_masks: dict[str, Tensor],
    event_data: dict[str, Tensor],
    sin_cos: bool,
    feature_scaler: FeatureScaler,
    var_helper: VariableHelper,
    train_type: Literal["particle", "event"],
    fs_mask_type: Literal["all", "charged", "neutral"] = "all",
) -> tuple[Tensor, Tensor]:
    """Build target (flow-sampling) data.

    In **particle mode** all *fs_vars* must come from one collection.
    In **event mode** variables are looked up from *event_data*
    (special handling for ``npflow`` / ``ntruth``).

    Args:
        fs_vars: Target variable names.
        collection_data: ``{collection: {bare_var: Tensor(…, N)}}``.
        collection_masks: ``{collection: BoolTensor(…, N)}``.
        event_data: ``{var_name: Tensor(…,)}`` for event-level variables.
        sin_cos: Apply sin/cos transform to phi.
        feature_scaler: :class:`FeatureScaler` instance.
        var_helper: :class:`VariableHelper`.
        train_type: ``"particle"`` or ``"event"``.
        fs_mask_type: ``"all"``, ``"charged"``, or ``"neutral"``.

    Returns:
        ``(fs_data, fs_mask)``:

        - particle: ``(…, N, F)`` and ``(…, N)``
        - event:    ``(…, F)`` and ``(…, N)`` all-False
    """
    features: list[Tensor] = []

    if train_type == "particle":
        # Determine target collection from first variable
        coll = var_helper.get_collection(fs_vars[0])
        if coll is None:
            raise ValueError(
                f"Target variable '{fs_vars[0]}' must have a collection prefix"
            )

        data = collection_data[coll]
        mask = collection_masks[coll].clone()

        for var in fs_vars:
            base = var_helper.get_base_name(var)
            x = data[base].float().unsqueeze(-1)  # (…, N) → (…, N, 1)
            transformed = apply_variable_transform(
                var, x, sin_cos, feature_scaler, var_helper
            )
            features.append(transformed)

        fs_data = torch.cat(features, dim=-1)  # (…, N, F)
        # Zero out padding (legacy pads AFTER transforms)
        fs_data = fs_data * collection_masks[coll].unsqueeze(-1).float()

        # Apply mask-type filtering for charged/neutral subsets
        if fs_mask_type != "all" and "class" in data:
            class_vals = data["class"]  # (…, N)
            if fs_mask_type == "charged":
                mask = mask & (class_vals < 3)
            elif fs_mask_type == "neutral":
                mask = mask & (class_vals >= 3)

        return fs_data.nan_to_num(0.0), mask

    # ---- event mode ----
    for var in fs_vars:
        # Check for n{collection} particle-count variables
        npart_coll = var_helper.is_npart_var(var)
        if npart_coll is not None:
            if npart_coll in collection_masks:
                x = collection_masks[npart_coll].sum(dim=-1).float()
            elif var in event_data:
                x = event_data[var].float()
            else:
                raise KeyError(
                    f"Cannot resolve '{var}': no '{npart_coll}' mask or event data"
                )
        else:
            if var not in event_data:
                raise KeyError(f"Variable '{var}' not found in event_data")
            x = event_data[var].float()

        x = x.unsqueeze(-1)  # (…,) → (…, 1)
        transformed = apply_variable_transform(
            var, x, sin_cos, feature_scaler, var_helper
        )
        features.append(transformed)

    fs_data = torch.cat(features, dim=-1)  # (…, F)

    # All-False mask matching any available collection mask shape
    any_mask = next(iter(collection_masks.values()))
    fs_mask = torch.zeros_like(any_mask, dtype=torch.bool)

    return fs_data, fs_mask


def build_ctxt_global_data(
    ctxt_global_vars: list[str],
    collection_masks: dict[str, Tensor],
    event_data: dict[str, Tensor],
    feature_scaler: FeatureScaler,
    var_helper: VariableHelper,
    means: Tensor | None = None,
) -> Tensor:
    """Build global context data (event-level features).

    Assembles scaled event-level variables.  ``ntruth`` / ``npflow`` are
    derived from ``collection_masks.sum(dim=-1)``.  If *means* is
    provided it is **prepended** (not appended) for backward compatibility
    with the ROOT backend.

    Args:
        ctxt_global_vars: e.g. ``["means", "pv_z", "truth_ht", "ntruth"]``.
        collection_masks: ``{collection: BoolTensor(…, N)}``.
        event_data: ``{var_name: Tensor(…,)}``.
        feature_scaler: :class:`FeatureScaler` instance.
        var_helper: :class:`VariableHelper`.
        means: Pre-computed means ``(…, n_mean_vars)``.  Prepended.

    Returns:
        Global context tensor:

        - Without means: ``(…, n_features)``
        - With means:    ``(…, n_mean_vars + n_features)``
    """
    features: list[Tensor] = []

    for var in ctxt_global_vars:
        if var == "means":
            continue  # Handled below — prepended at the end

        # Check for n{collection} particle-count variables (ntruth, npflow, etc.)
        npart_coll = var_helper.is_npart_var(var)
        if npart_coll is not None:
            if npart_coll not in collection_masks:
                raise ValueError(
                    f"Cannot compute {var}: '{npart_coll}' not in collection_masks"
                )
            x = collection_masks[npart_coll].sum(dim=-1).float().unsqueeze(-1)
            features.append(feature_scaler["npart"].transform(x).float())

        else:
            if var not in event_data:
                raise KeyError(f"Global variable '{var}' not found in event_data")
            x = event_data[var].float().unsqueeze(-1)  # (…,) → (…, 1)
            scaler_key = var_helper.scaler_key(var)
            features.append(feature_scaler[scaler_key].transform(x).float())

    # Concatenate features
    if features:
        global_data = torch.cat(features, dim=-1)  # (…, n_features)
    else:
        if means is not None:
            batch_shape = means.shape[:-1]
        elif collection_masks:
            batch_shape = next(iter(collection_masks.values())).shape[:-1]
        else:
            batch_shape = ()
        global_data = torch.zeros(*batch_shape, 0, dtype=torch.float32)

    # Prepend means if provided (ROOT convention: means FIRST)
    if means is not None:
        global_data = torch.cat([means.float(), global_data], dim=-1)

    return global_data


# ---------------------------------------------------------------------------
# Means computation — shape-agnostic
# ---------------------------------------------------------------------------


def compute_means(
    collection_data: dict[str, Tensor],
    ctxt_vars: list[str],
    mask: Tensor,
    var_helper: VariableHelper,
    collection_name: str,
) -> Tensor:
    """Compute per-event means of context variables.

    Shape-agnostic: works on single events ``(…, N)`` and batches ``(B, N)``.
    Excludes variables containing ``class`` or ``charge``.

    For bulk init-time computation over the entire dataset (ROOT),
    see :meth:`ComputedVariables.compute_flat_means` which uses
    ``np.add.reduceat`` for efficiency.

    Args:
        collection_data: ``{bare_var: Tensor(…, N)}`` for target collection.
        ctxt_vars: Context variable names **with** collection prefixes.
        mask: Bool tensor ``(…, N)`` — True for valid particles.
        var_helper: :class:`VariableHelper` for prefix handling.
        collection_name: Collection to filter vars by (e.g. ``"truth"``).

    Returns:
        Tensor ``(…, n_vars)`` with per-event means.
    """
    filtered = [
        v
        for v in ctxt_vars
        if var_helper.get_collection(v) == collection_name
        and "class" not in v
        and "charge" not in v
    ]

    if not filtered:
        batch_shape = mask.shape[:-1]
        return torch.zeros(*batch_shape, 0, dtype=torch.float32, device=mask.device)

    # Stack variables: (…, N, n_vars)
    stacked = torch.stack(
        [collection_data[var_helper.get_base_name(v)].float() for v in filtered],
        dim=-1,
    )

    # Masked mean: zero out padding, divide by valid particle count
    masked = stacked * mask.unsqueeze(-1).float()  # (…, N, n_vars)
    counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1)  # (…, 1)
    means = masked.sum(dim=-2) / counts  # (…, n_vars)

    return means
