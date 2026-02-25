"""Variable transformation utilities for physics datasets.

Handles:
- Sin/cos transforms for angular variables (phi)
- One-hot encoding for categorical variables (class, charge)
- Feature scaling via FeatureScaler with 3-step scaler key logic

All transforms are shape-agnostic: they work on any leading batch dimensions.
Input shape ``(..., 1)``, output shape ``(..., K)`` where K depends on
transform type.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from parnassus_core.data.utils.variable_helper import VariableHelper
from parnassus_core.data.utils.scaling import FeatureScaler


def apply_variable_transform(
    var_name: str,
    data: Tensor,
    sin_cos: bool,
    feature_scaler: FeatureScaler,
    var_helper: VariableHelper | None = None,
) -> Tensor:
    """Apply appropriate transform to a variable based on its name.

    Shape-agnostic: works on any leading dimensions. Input ``(..., 1)``,
    output ``(..., K)`` where K depends on the transform type.

    Transformation logic:
    1. phi variables + sin_cos=True → ``[sin(phi), cos(phi)]`` → ``(..., 2)``
    2. charge variables → one-hot (3 classes: -1, 0, +1) → ``(..., 3)``
    3. class variables → one-hot (5 classes: 0-4) → ``(..., 5)``
    4. All others → feature scaling → ``(..., 1)``

    For feature scaling, the scaler key is derived via
    ``VariableHelper.scaler_key()`` (3-step: strip prefix → normalize
    npart → strip "My").

    Args:
        var_name: Variable name (with or without collection prefix)
        data: Input tensor, shape ``(..., 1)``
        sin_cos: Whether to apply sin/cos transform to phi variables
        feature_scaler: FeatureScaler instance for numerical scaling
        var_helper: VariableHelper for scaler key derivation. If None,
            falls back to static 3-step logic with hardcoded prefixes.

    Returns:
        Transformed tensor, shape ``(..., K)`` varies by transform type:
        - sin/cos: ``(..., 2)``
        - one-hot charge: ``(..., 3)``
        - one-hot class: ``(..., 5)``
        - scaled: ``(..., 1)``

    Examples:
        >>> vh = VariableHelper(["truth", "pflow"])
        >>> phi = torch.tensor([[1.5], [2.0], [3.0]])
        >>> result = apply_variable_transform("truth_phi", phi, True, scaler, vh)
        >>> result.shape
        torch.Size([3, 2])  # [sin(phi), cos(phi)]

        >>> cls = torch.tensor([[0], [2], [4]])
        >>> result = apply_variable_transform("pflow_class", cls, False, scaler, vh)
        >>> result.shape
        torch.Size([3, 5])  # one-hot encoded
    """
    # Ensure input is float32 for consistency
    data = data.float()

    # 1. Sin/cos transform for angular variables
    if var_name.endswith("phi") and sin_cos:
        return torch.cat([torch.sin(data), torch.cos(data)], dim=-1)

    # 2. One-hot encoding for charge (-1, 0, +1 → 0, 1, 2)
    if var_name.endswith("charge"):
        # squeeze(-1): (..., 1) → (...,) — safe for any batch dimension
        return F.one_hot(data.long().squeeze(-1) + 1, num_classes=3).float()

    # 3. One-hot encoding for class (0-4 → 5 classes)
    if var_name.endswith("class"):
        # squeeze(-1): (..., 1) → (...,) — safe for any batch dimension
        return F.one_hot(data.long().squeeze(-1), num_classes=5).float()

    # 4. Feature scaling for all other variables
    if var_helper is None:
        raise ValueError(f"var_helper is required to scale '{var_name}'")
    scaler_key = var_helper.scaler_key(var_name)
    return feature_scaler[scaler_key].transform(data).float()


def apply_batch_transform(
    var_names: list[str],
    data_dict: dict[str, Tensor],
    sin_cos: bool,
    feature_scaler: FeatureScaler,
    var_helper: VariableHelper | None = None,
) -> list[Tensor]:
    """Apply transforms to multiple variables and return as list.

    Args:
        var_names: List of variable names to transform
        data_dict: Dictionary mapping variable names to tensors
        sin_cos: Whether to apply sin/cos transform
        feature_scaler: FeatureScaler instance
        var_helper: VariableHelper for scaler key derivation

    Returns:
        List of transformed tensors (to be concatenated by caller)

    Example:
        >>> var_names = ["truth_pt", "truth_eta", "truth_phi"]
        >>> data_dict = {
        ...     "truth_pt": torch.tensor([[10.0]]),
        ...     "truth_eta": torch.tensor([[1.5]]),
        ...     "truth_phi": torch.tensor([[2.0]]),
        ... }
        >>> result = apply_batch_transform(var_names, data_dict, True, scaler)
        >>> len(result)
        3
        >>> result[2].shape  # phi with sin_cos=True
        torch.Size([1, 2])
    """
    transformed = []
    for var_name in var_names:
        data = data_dict[var_name]
        transformed.append(
            apply_variable_transform(
                var_name, data, sin_cos, feature_scaler, var_helper
            )
        )
    return transformed
