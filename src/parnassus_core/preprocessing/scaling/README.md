# Scaling Config for calculate_transforms_batched.py

This folder contains:
- calculate_transforms_batched.py
- cms_default_config.yaml

The script uses a YAML config to define:
- which event and particle variables to process
- how each variable is treated (deferred/charged/common)
- optional transform function and scale per variable
- whether event-level pv_x/pv_y/pv_z are recalculated from truth vertices

## Run

```bash
python -m parnassus_core.preprocessing.scaling.calculate_transforms_batched \
    -i /path/to/input.h5 \
    -c cms_default_config.yaml \
    -o /path/to/var_transforms.yaml
```

Additional arguments:
- `--batch-size`: number of events to process per batch (default: 100,000)
- `--max-events`: maximum number of events to process (default: None, i.e. all)
- `--skip-particles`: if set, only processes event-level variables and skips particle-level transforms
- `--split-class`: if set, computes separate particle stats for each class from `settings.class_ids`
- `--recalculate-pv`: if set, overrides config and recalculates event `pv_x`, `pv_y`, `pv_z`

## Config Structure

Top-level sections:
- event_vars
- particle_vars
- settings

### event_vars / particle_vars

Each variable name maps to the same structure:

- `fn`: optional transform function name (`arcsinh_iqr`, `arcsinh`, `log_median`, ...)
- `fn_scale`: optional transform scale
- `deferred`: boolean; if true, variable is processed in deferred pass
- `charged`: boolean; if true, only charged particles are used
- `common`: boolean; if true, combine truth and pflow values

Supported extra optional key:
- `fn_shift`: transform shift (used for log-style transforms)

Auto-filled behavior for deferred transform functions:
- For `fn: arcsinh_iqr`, median/iqr are estimated from deferred data.
- For `fn: log_median`, median is estimated from deferred data.

### settings

Optional run settings (defaults shown):
- `class_ids`: `[0, 1, 2, 3, 4]`
- `charged_class_lt`: `3`
- `recalculate_pv`: `false`

Notes:
- If `recalculate_pv: true` and `event_vars` contains `pv_x`/`pv_y`/`pv_z`, these values are computed per event from the most frequent valid truth-vertex coordinates.
- Recalculation always uses `truth_vx`, `truth_vy`, `truth_vz` (read from truth particle branches `vx`, `vy`, `vz`).
- `vz_shifted` is fixed and always computed as `vz - pv_z`.

## Minimal Example

    event_vars:
      ht: {common: true}
      met_x: {common: true}
      met_y: {common: true}
      npart: {common: true}
      pv_x: {}
      pv_y: {}
      pv_z: {}
      actualNumPU: {}
      trueNumPU: {}

    particle_vars:
      pt: {common: true}
      eta: {common: true}
      phi: {common: true}
      ptrel: {common: true}
      class: {common: true}

      vx: {fn: arcsinh_iqr, deferred: true, charged: true}
      vy: {fn: arcsinh_iqr, deferred: true, charged: true}
      vz_shifted: {fn: arcsinh, fn_scale: 0.1, deferred: true, charged: true}
      d0: {fn: arcsinh, fn_scale: 0.1, deferred: true, charged: true}
      d0Error: {fn: log_median, fn_shift: 1.0e-6, deferred: true, charged: true}
      z0: {fn: arcsinh, fn_scale: 0.1, deferred: true, charged: true}
      z0Error: {fn: log_median, fn_shift: 1.0e-6, deferred: true, charged: true}

    settings:
      class_ids: [0, 1, 2, 3, 4]
      charged_class_lt: 3
      recalculate_pv: true

## Output Format

The script generates a YAML file with statistics for each variable:

```yaml
event_vars:
  truth_ht:
    mean: 450.123
    std: 120.456
    min: 0.0
    max: 2500.0
  ...

particle_vars:
  pt:
    mean: 15.234
    std: 25.678
    min: 0.0
    max: 500.0
  d0:
    mean: 0.001
    std: 0.012
    min: -0.05
    max: 0.05
    transform_fn: arctan
    transform_scale: 0.05
  ...
```
