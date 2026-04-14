from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from .config import VariableConfig
else:
    VariableConfig = Any


class OnlineStats:
    """Calculate running statistics for batched data."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        self.min_val = np.inf
        self.max_val = -np.inf

    def update(self, new_data: np.ndarray):
        """Update statistics with a new batch of data using vectorized Chan's method."""
        new_data = new_data.flatten()
        new_data = new_data[~np.isnan(new_data)]  # Remove NaNs

        n_b = len(new_data)
        if n_b == 0:
            return

        # 1. Vectorized Update for Min/Max
        self.min_val = min(self.min_val, np.min(new_data))
        self.max_val = max(self.max_val, np.max(new_data))

        # 2. Calculate batch statistics
        mean_b = np.mean(new_data)
        m2_b = np.var(new_data) * n_b
        n_a = self.count

        # 3. Merge batch stats into global stats
        total_count = n_a + n_b
        delta = mean_b - self.mean
        self.mean += delta * (n_b / total_count)
        self.m2 += m2_b + (delta**2) * n_a * n_b / total_count
        self.count = total_count

    @property
    def std(self) -> float:
        """Return standard deviation."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.m2 / self.count)

    def get_stats(self) -> dict:
        """Return statistics as dictionary."""
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min_val),
            "max": float(self.max_val),
        }


def apply_transform(
    arr: np.ndarray,
    fn: Optional[str] = None,
    fn_scale: float = 1.0,
    fn_shift: float = 0.0,
) -> np.ndarray:
    """Apply transformation to array (deprecated, use apply_custom_transform)."""
    arr = (arr + fn_shift) / fn_scale
    if fn is not None:
        arr = getattr(np, fn)(arr)
    return arr


def apply_custom_transform(arr: np.ndarray, config: dict) -> np.ndarray:
    """Apply custom transformation based on config."""
    fn = config.get("fn", None)

    if fn is None:
        return arr

    abs_clip = config.get("abs_clip", None)
    if abs_clip is not None:
        mask = np.abs(arr) < abs_clip
    else:
        mask = np.ones_like(arr, dtype=bool)

    if fn == "arcsinh_iqr":
        median = config["median"]
        iqr = config["iqr"]
        return np.arcsinh((arr - median) / iqr)[mask]

    if fn == "log_median":
        fn_shift = config.get("fn_shift", 0.0)
        median = config.get("median", 1.0)
        return np.log((arr + fn_shift) / median)[mask]

    return apply_transform(
        arr,
        fn=fn,
        fn_scale=config.get("fn_scale", 1.0),
        fn_shift=config.get("fn_shift", 0.0),
    )[mask]


def get_deferred_configs(
    particle_var_configs: dict[str, VariableConfig],
    deferred_vars: list[str],
    vtx_params: dict[str, dict[str, float]],
    error_medians: dict[str, float],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    for var_name in deferred_vars:
        var_cfg = particle_var_configs[var_name]
        config: dict[str, Any] = {}

        if var_cfg.fn is not None:
            config["fn"] = var_cfg.fn
        if var_cfg.fn_scale is not None:
            config["fn_scale"] = var_cfg.fn_scale
        if var_cfg.fn_shift != 0.0:
            config["fn_shift"] = var_cfg.fn_shift
        if var_cfg.abs_clip is not None:
            config["abs_clip"] = var_cfg.abs_clip

        if var_cfg.fn == "arcsinh_iqr":
            config["median"] = vtx_params.get(var_name, {}).get("median", 0.0)
            config["iqr"] = (
                var_cfg.fn_scale
                if var_cfg.fn_scale is not None
                else vtx_params.get(var_name, {}).get("iqr", 1.0)
            )
        elif var_cfg.fn == "log_median":
            config["median"] = error_medians.get(var_name, 0.001)

        output[var_name] = config

    return output
