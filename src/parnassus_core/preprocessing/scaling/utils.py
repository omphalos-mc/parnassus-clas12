from typing import Any, Optional

import numpy as np


def round_floats(data, digits: int = 4):
    """Recursively round float values in nested structures."""
    if isinstance(data, dict):
        return {k: round_floats(v, digits) for k, v in data.items()}
    if isinstance(data, list):
        return [round_floats(v, digits) for v in data]
    if isinstance(data, float):
        if abs(data) < 1e-4:
            return data  # Don't round very small numbers
        return round(data, digits)
    return data


def normalize_dump_convention(data):
    """Normalize transform keys to the var_transform_new.yaml convention."""
    if isinstance(data, dict):
        normalized = {k: normalize_dump_convention(v) for k, v in data.items()}

        if "transform_fn" not in normalized:
            return normalized

        transform_fn = normalized.pop("transform_fn")
        if "transform_scale" in normalized:
            normalized["fn_scale"] = normalized.pop("transform_scale")
        if "transform_shift" in normalized:
            normalized["fn_shift"] = normalized.pop("transform_shift")

        if transform_fn == "log_median":
            normalized["fn"] = "log"
            if "median" in normalized:
                normalized["fn_scale"] = normalized.pop("median")
        elif transform_fn == "arcsinh_iqr":
            normalized["fn"] = "asinh"
            median = normalized.pop("median", None)
            iqr = normalized.pop("iqr", None)
            if iqr is not None:
                normalized["fn_scale"] = iqr
            if median is not None:
                normalized["fn_shift"] = -median
        elif transform_fn == "arcsinh":
            normalized["fn"] = "asinh"
        else:
            normalized["fn"] = transform_fn

        return normalized

    if isinstance(data, list):
        return [normalize_dump_convention(v) for v in data]

    return data


def get_pv_coords(truth_vertex: np.ndarray) -> np.ndarray:
    if len(truth_vertex) == 0:
        return np.array(0.0)
    values, counts = np.unique(truth_vertex, return_counts=True)
    pv_index = np.argmax(counts)
    pv_coords = values[pv_index]
    return pv_coords


def get_batch_pv_data(
    truth_data: np.ndarray, truth_valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute per-event PV coordinates for x/y/z from truth vertices."""
    truth_names = truth_data.dtype.names or ()
    required = ("vx", "vy", "vz")
    missing = [var for var in required if var not in truth_names]
    if missing:
        raise ValueError(
            f"Truth variables {missing} are required for PV recalculation but not found in dataset"
        )

    pv_data = {}
    for var in required:
        pv_data[var] = np.array(
            [
                get_pv_coords(truth_data[var][i][truth_valid[i]])
                for i in range(len(truth_data))
            ]
        )

    return pv_data


def resolve_particle_bases(
    var: str,
    var_cfg,
    pflow_data: np.ndarray,
    truth_data: np.ndarray,
    pflow_names: tuple[str, ...] | list[str],
    truth_names: tuple[str, ...] | list[str],
    pflow_ptrel: np.ndarray,
    truth_ptrel: np.ndarray,
    pv_data: dict[str, np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Resolve per-source arrays for a particle variable before masking."""
    if var == "ptrel":
        return pflow_ptrel, truth_ptrel if var_cfg.common else None

    if var == "vz_shifted":
        if "vz" not in pv_data:
            raise ValueError("PV data for 'vz' is required for 'vz_shifted'")

        pflow_base = (
            pflow_data["vz"] - pv_data["vz"][:, np.newaxis]
            if "vz" in pflow_names
            else None
        )
        truth_base = (
            truth_data["vz"] - pv_data["vz"][:, np.newaxis]
            if var_cfg.common and "vz" in truth_names
            else None
        )
        return pflow_base, truth_base

    pflow_base = pflow_data[var] if var in pflow_names else None
    truth_base = truth_data[var] if var_cfg.common and var in truth_names else None
    return pflow_base, truth_base


def combine_sources_with_classes(
    pflow_base: Optional[np.ndarray],
    truth_base: Optional[np.ndarray],
    pflow_mask: np.ndarray,
    truth_mask: np.ndarray,
    pflow_classes: np.ndarray,
    truth_classes: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build flattened combined values and classes from available sources."""
    arr_list = []
    class_list = []

    if pflow_base is not None:
        pflow_values = pflow_base[pflow_mask]
        if len(pflow_values) > 0:
            arr_list.append(pflow_values)
            class_list.append(pflow_classes[pflow_mask])

    if truth_base is not None:
        truth_values = truth_base[truth_mask]
        if len(truth_values) > 0:
            arr_list.append(truth_values)
            class_list.append(truth_classes[truth_mask])

    if not arr_list:
        return None, None

    return np.concatenate(arr_list), np.concatenate(class_list)


def iter_class_slices(
    combined: np.ndarray, combined_class: np.ndarray, class_ids: list[int]
):
    """Yield per-class slices from already-combined arrays."""
    for class_id in class_ids:
        class_mask = combined_class == class_id
        if np.any(class_mask):
            yield class_id, combined[class_mask]


def attach_transform_metadata(target: dict[str, Any], config: dict[str, Any]) -> None:
    """Attach transform metadata fields to an output stats dict."""
    field_map = {
        "fn": "transform_fn",
        "fn_scale": "transform_scale",
        "fn_shift": "transform_shift",
        "median": "median",
        "iqr": "iqr",
    }
    for source_key, target_key in field_map.items():
        if source_key in config:
            target[target_key] = config[source_key]


def print_stats_block(var_name: str, values: dict[str, Any]) -> None:
    """Print a variable stats block with consistent formatting."""
    print(f"{var_name}:")
    for key, value in values.items():
        if isinstance(value, str):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.3f}")
