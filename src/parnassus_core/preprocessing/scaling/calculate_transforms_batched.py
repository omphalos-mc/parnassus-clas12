"""
Calculate variable transformations using batched loading.

This script computes mean, std, min, and max for event and particle variables
by processing data in batches to avoid memory issues with large datasets.

Transformations applied (from check_scaling.ipynb):
- Vertex vars (vx, vy, vz): arcsinh((x - median) / IQR)
- Impact params (d0, z0): arcsinh(10 * x)
- Impact errors (d0Error, z0Error): log((x + 1e-6) / median)

Uses a single-pass algorithm for particle stats:
1. During pass: Calculate stats for simple variables, collect vertex/impact data
2. After pass: Calculate IQR/median, apply transformations, compute final stats
"""

import argparse
from pathlib import Path
from typing import Optional, cast

import h5py

# Import hdf5plugin for compression support (side-effect import)
import hdf5plugin  # noqa: F401
import numpy as np
import yaml
from tqdm import tqdm

from parnassus_core.preprocessing.scaling.config import (
    RuntimeConfig,
)
from parnassus_core.preprocessing.scaling.transforms import (
    OnlineStats,
    apply_custom_transform,
    get_deferred_configs,
)
from parnassus_core.preprocessing.scaling.utils import (
    attach_transform_metadata,
    combine_sources_with_classes,
    get_batch_pv_data,
    get_pv_coords,
    iter_class_slices,
    normalize_dump_convention,
    print_stats_block,
    resolve_particle_bases,
    round_floats,
)


def calculate_event_stats(
    file_path: str,
    runtime_config: RuntimeConfig,
    batch_size: int = 100_000,
    max_events: Optional[int] = None,
    precomputed_pv_stats: Optional[dict[str, dict[str, float]]] = None,
) -> dict:
    """Calculate statistics for event-level variables."""
    print("Calculating event-level statistics...")

    # Initialize stats collectors for each variable
    event_var_configs = runtime_config.event_vars
    event_vars = list(event_var_configs.keys())
    stats = {var: OnlineStats() for var in event_vars}
    recalculate_vars = {"pv_x": "vx", "pv_y": "vy", "pv_z": "vz"}
    precomputed_pv_stats = precomputed_pv_stats or {}

    # Open file once and keep it open
    with h5py.File(file_path, "r") as f:
        events_dataset = cast(h5py.Dataset, f["events"])
        truth_dataset: Optional[h5py.Dataset] = None
        need_runtime_pv_recalc = runtime_config.recalculate_pv and any(
            var_name in recalculate_vars and var_name not in precomputed_pv_stats
            for var_name in event_vars
        )
        if need_runtime_pv_recalc:
            truth_dataset = cast(h5py.Dataset, f["truth"])

        total_events = len(events_dataset) if max_events is None else max_events
        n_batches = (total_events + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Event batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_events)

            event_data = events_dataset[start_idx:end_idx]
            event_names = event_data.dtype.names or ()
            truth_data = None
            truth_valid = None
            if truth_dataset is not None:
                truth_data = truth_dataset[start_idx:end_idx]
                truth_valid = truth_data["valid"] > 0

            # Update statistics
            for key in event_vars:
                var_cfg = event_var_configs[key]
                if key in precomputed_pv_stats:
                    continue

                if runtime_config.recalculate_pv and key in recalculate_vars:
                    assert truth_data is not None
                    assert truth_valid is not None
                    truth_key = recalculate_vars[key]
                    pv_coords = np.array(
                        [
                            get_pv_coords(truth_data[truth_key][i][truth_valid[i]])
                            for i in range(len(truth_data))
                        ]
                    )
                    stats[key].update(pv_coords)
                    continue

                if key in event_names:
                    stats[key].update(event_data[key])
                if var_cfg.common:
                    truth_key = f"truth_{key}"
                    pflow_key = f"pflow_{key}"
                    if truth_key in event_names:
                        stats[key].update(event_data[truth_key])
                    if pflow_key in event_names:
                        stats[key].update(event_data[pflow_key])

    # Convert to output format
    output = {}
    for var, stat_obj in stats.items():
        if var in precomputed_pv_stats:
            output[var] = precomputed_pv_stats[var]
            print_stats_block(var, output[var])
            continue

        if stat_obj.count == 0:
            raise ValueError(f"No values found for event variable '{var}'")
        output[var] = stat_obj.get_stats()
        print_stats_block(var, output[var])

    return output


def calculate_particle_stats(
    file_path: str,
    runtime_config: RuntimeConfig,
    batch_size: int = 100_000,
    max_events: Optional[int] = None,
    split_class: bool = False,
) -> tuple[dict, dict[str, dict[str, float]]]:
    """Calculate statistics for particle-level variables with proper transformations."""
    print("\nCalculating particle-level statistics...")

    # Variables that need special treatment (need IQR/median before transform)
    particle_var_configs = runtime_config.particle_vars
    deferred_vars = [
        var_name for var_name, cfg in particle_var_configs.items() if cfg.deferred
    ]
    immediate_vars = [
        var_name for var_name, cfg in particle_var_configs.items() if not cfg.deferred
    ]

    charged_class_lt = runtime_config.charged_class_lt

    # Storage for deferred variables (collect raw data)
    deferred_data = {var: [] for var in deferred_vars}

    class_ids = runtime_config.class_ids
    class_stats = None
    deferred_data_by_class = None
    if split_class:
        class_stats = {
            class_id: {var: OnlineStats() for var in immediate_vars}
            for class_id in class_ids
        }
        deferred_data_by_class = {
            class_id: {var: [] for var in deferred_vars} for class_id in class_ids
        }

    # Initialize stats collectors for immediate variables
    stats = {var: OnlineStats() for var in immediate_vars}
    pv_event_stats = {}
    if runtime_config.recalculate_pv:
        pv_event_stats = {
            "pv_x": OnlineStats(),
            "pv_y": OnlineStats(),
            "pv_z": OnlineStats(),
        }

    # Single pass: calculate immediate stats and collect deferred data
    print("Processing batches...")
    with h5py.File(file_path, "r") as f:
        pflow_dataset = cast(h5py.Dataset, f["pflow"])
        truth_dataset = cast(h5py.Dataset, f["truth"])
        pflow_data = np.array(0, dtype=pflow_dataset.dtype)
        truth_data = np.array(0, dtype=truth_dataset.dtype)

        total_events = len(pflow_dataset) if max_events is None else max_events
        n_batches = (total_events + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Particle batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_events)
            pflow_data.resize(
                (end_idx - start_idx, *pflow_dataset.shape[1:]), refcheck=False
            )
            truth_data.resize(
                (end_idx - start_idx, *truth_dataset.shape[1:]), refcheck=False
            )

            pflow_dataset.read_direct(pflow_data, np.s_[start_idx:end_idx])
            truth_dataset.read_direct(truth_data, np.s_[start_idx:end_idx])

            # Calculate ptrel for pflow and truth
            pflow_pt = pflow_data["pt"]
            truth_pt = truth_data["pt"]
            pflow_ptrel = pflow_pt / np.nansum(pflow_pt, axis=-1, keepdims=True)
            truth_ptrel = truth_pt / np.nansum(truth_pt, axis=-1, keepdims=True)

            # Process all particle variables in one loop.
            pflow_valid = pflow_data["valid"] > 0
            truth_valid = truth_data["valid"] > 0
            pflow_names = pflow_data.dtype.names or []
            truth_names = truth_data.dtype.names or []
            need_pv_data = (
                runtime_config.recalculate_pv or "vz_shifted" in deferred_vars
            )
            pv_data = {}
            if need_pv_data:
                pv_data = get_batch_pv_data(truth_data, truth_valid)

            if runtime_config.recalculate_pv:
                for var in ["vx", "vy", "vz"]:
                    pv_event_stats[f"pv_{var[-1]}"].update(pv_data[var])

            for var, var_cfg in particle_var_configs.items():
                is_deferred = var_cfg.deferred
                pflow_mask = pflow_valid.copy()
                truth_mask = truth_valid.copy()
                if var_cfg.charged:
                    pflow_mask &= pflow_data["class"] < charged_class_lt
                    truth_mask &= truth_data["class"] < charged_class_lt

                pflow_base, truth_base = resolve_particle_bases(
                    var,
                    var_cfg,
                    pflow_data,
                    truth_data,
                    pflow_names,
                    truth_names,
                    pflow_ptrel,
                    truth_ptrel,
                    pv_data,
                )

                combined, combined_class = combine_sources_with_classes(
                    pflow_base,
                    truth_base,
                    pflow_mask,
                    truth_mask,
                    pflow_data["class"],
                    truth_data["class"],
                )

                if combined is None or combined_class is None:
                    continue

                if is_deferred:
                    deferred_data[var].append(combined)
                    if split_class and deferred_data_by_class is not None:
                        allowed_class_ids = [
                            class_id
                            for class_id in class_ids
                            if not (var_cfg.charged and class_id >= charged_class_lt)
                        ]
                        for class_id, class_values in iter_class_slices(
                            combined, combined_class, allowed_class_ids
                        ):
                            deferred_data_by_class[class_id][var].append(class_values)
                else:
                    stats[var].update(combined)
                    if split_class and class_stats is not None:
                        for class_id, class_values in iter_class_slices(
                            combined, combined_class, class_ids
                        ):
                            class_stats[class_id][var].update(class_values)

    # Calculate IQR/median for deferred variables
    print("\nCalculating IQR and medians for vertex/impact variables...")

    vertex_source_vars = {
        var for var in deferred_vars if particle_var_configs[var].fn == "arcsinh_iqr"
    }
    error_source_vars = {
        var for var in deferred_vars if particle_var_configs[var].fn == "log_median"
    }

    # Vertex variables: need IQR and median
    vertex_params = {}
    for var in deferred_vars:
        if var not in vertex_source_vars:
            continue
        if deferred_data[var]:
            var_all = np.concatenate(deferred_data[var])
            median = float(np.median(var_all))
            iqr = float(np.percentile(var_all, 75) - np.percentile(var_all, 25))
            vertex_params[var] = {"median": median, "iqr": iqr}
            print(f"  {var}: median={median:.5f}, iqr={iqr:.5f}")

    # Error variables: need median
    error_medians = {}
    for var in deferred_vars:
        if var not in error_source_vars:
            continue
        if deferred_data[var]:
            var_all = np.concatenate(deferred_data[var])
            median = float(np.median(var_all))
            error_medians[var] = median
            print(f"  {var}: median={median:.6f}")

    # Now calculate stats for deferred variables with transformations
    print("\nCalculating statistics for vertex/impact variables...")

    # Define transformations for deferred variables
    deferred_configs = get_deferred_configs(
        particle_var_configs,
        deferred_vars,
        vertex_params,
        error_medians,
    )

    # Calculate stats for each deferred variable
    for var in deferred_vars:
        if deferred_data[var]:
            var_all = np.concatenate(deferred_data[var])
            # Apply transformation
            var_transformed = apply_custom_transform(var_all, deferred_configs[var])
            # Calculate stats
            stats[var] = OnlineStats()
            stats[var].update(var_transformed)

    # Convert to output format
    output = {}
    for var, stat_obj in stats.items():
        output[var] = stat_obj.get_stats()

        # Add transformation info for deferred variables
        if var in deferred_configs:
            attach_transform_metadata(output[var], deferred_configs[var])

        print_stats_block(var, output[var])

    if split_class and class_stats is not None and deferred_data_by_class is not None:
        print("\nCalculating class-split particle statistics...")
        class_output = {}

        for class_id in class_ids:
            per_class = {}

            for var, stat_obj in class_stats[class_id].items():
                if stat_obj.count > 0:
                    per_class[var] = stat_obj.get_stats()

            vertex_params_class = {}
            for var in deferred_vars:
                if var not in vertex_source_vars:
                    continue
                class_data = deferred_data_by_class[class_id][var]
                if class_data:
                    var_all = np.concatenate(class_data)
                    var_all = var_all[~np.isnan(var_all)]
                    if len(var_all) > 0:
                        median = float(np.median(var_all))
                        iqr = float(
                            np.percentile(var_all, 75) - np.percentile(var_all, 25)
                        )
                        vertex_params_class[var] = {"median": median, "iqr": iqr}

            error_medians_class = {}
            for var in deferred_vars:
                if var not in error_source_vars:
                    continue
                class_data = deferred_data_by_class[class_id][var]
                if class_data:
                    var_all = np.concatenate(class_data)
                    var_all = var_all[~np.isnan(var_all)]
                    if len(var_all) > 0:
                        error_medians_class[var] = float(np.median(var_all))

            deferred_configs_class = get_deferred_configs(
                particle_var_configs,
                deferred_vars,
                vertex_params_class,
                error_medians_class,
            )

            for var in deferred_vars:
                class_data = deferred_data_by_class[class_id][var]
                if class_data:
                    var_all = np.concatenate(class_data)
                    var_all = var_all[~np.isnan(var_all)]
                    if len(var_all) == 0:
                        continue
                    var_transformed = apply_custom_transform(
                        var_all, deferred_configs_class[var]
                    )
                    var_stat = OnlineStats()
                    var_stat.update(var_transformed)
                    per_class[var] = var_stat.get_stats()

                    attach_transform_metadata(
                        per_class[var], deferred_configs_class[var]
                    )

            class_output[f"class_{class_id}"] = per_class

        output["by_class"] = class_output

    pv_event_output = {}
    for var_name, stat_obj in pv_event_stats.items():
        if stat_obj.count > 0:
            pv_event_output[var_name] = stat_obj.get_stats()

    return output, pv_event_output


def main():
    parser = argparse.ArgumentParser(
        description="Calculate variable transformations using batched loading"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input HDF5 file",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML config with variables and transform parameters",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="var_transforms.yaml",
        help="Output YAML file for transformations",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of events to process per batch",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process (default: all)",
    )
    parser.add_argument(
        "--skip-particles",
        action="store_true",
        help="Skip particle-level statistics (faster)",
    )
    parser.add_argument(
        "--split-class",
        action="store_true",
        help="Split particle-level statistics by particle class (0..4)",
    )
    parser.add_argument(
        "--recalculate-pv",
        action="store_true",
        help="Recalculate event pv_x/pv_y/pv_z from truth vertices",
    )

    args = parser.parse_args()

    config = RuntimeConfig.from_yaml(args.config)
    if args.recalculate_pv:
        config.recalculate_pv = True

    if not args.skip_particles:
        particle_stats, precomputed_pv_stats = calculate_particle_stats(
            args.input,
            runtime_config=config,
            batch_size=args.batch_size,
            max_events=args.max_events,
            split_class=args.split_class,
        )
    else:
        particle_stats = {}
        precomputed_pv_stats = {}

    # Calculate statistics
    event_stats = calculate_event_stats(
        args.input,
        runtime_config=config,
        batch_size=args.batch_size,
        max_events=args.max_events,
        precomputed_pv_stats=precomputed_pv_stats,
    )

    # Combine and save
    output_data = {
        "event_vars": event_stats,
        "particle_vars": particle_stats,
    }
    output_data = normalize_dump_convention(output_data)
    output_data = round_floats(output_data, digits=4)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nTransformations saved to {output_path}")


if __name__ == "__main__":
    main()
