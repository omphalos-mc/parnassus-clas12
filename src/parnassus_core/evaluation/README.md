# Evaluation Module

This folder contains the full inference and postprocessing pipeline used to:
- generate model predictions from trained checkpoints
- save them to ROOT files
- convert those outputs into plotting-ready trees with jets and optional matching

## Main files in this folder

- eval.py
	Runs inference for all datasets listed in an evaluation config.
- event_filter.py
	The scatE pre-filter: the classifier and the event selection eval.py applies while reading.
- postprocess_eval.py
	Postprocesses evaluation ROOT output: masks particles, clusters jets, runs optional matching, writes final ROOT output.

## Recommended workflow

1. Create an evaluation config (same structure as analysis.yaml).
2. Create a variable config (same structure as eval_vars.yaml).
3. Run eval.py.
4. Run postprocess_eval.py on eval output.

## 1) Evaluation Config Structure

Reference example:
- [src/parnassus_core/evaluation/example_configs/eval_config.yaml](./example_configs/eval_config.yaml)

Required top-level fields:

- `eval_path`: Output directory for eval ROOT files. Created if missing.
- `recalculate_ip`: Boolean. If true and impact model is enabled, d0 and z0 approximations are computed from generated kinematics and PV before impact generation.
- `scale_dict`: Path to scaling YAML used by `FeatureScaler` and `Unscaler` during generation. Should be the same file used during training.
- `var_dict`: Path to variable config YAML (described in section 2).
- `data`
	Mapping with:
	- `top_folder`: prefix directory used for dataset file paths
	- `max_particles`: max sequence length for EagerDataset
	- `eta_clamp`: optional `[min, max]` pair that eta is clipped to when loading (default `[-3.0, 3.0]`, the CMS range). **Must match the `eta_clamp` used in training** — the clamp reassigns out-of-range eta rather than dropping particles, so a mismatch silently feeds out-of-distribution conditioning to the models. Set `null` to disable clipping. CLAS12 uses `[-3.0, 5.0]`.
	- `class_based_scaling`: optional boolean (default `false`). Scales per-particle variables with the `class_N` / `pdgId_N` blocks of `scale_dict`, for both the truth conditioning and the unscaling of generated particles. **Must match `data.class_based_scaling` in the training config** — a mismatch silently mis-scales both.
	- `datasets`: list of dataset definitions
        Each data.datasets entry:
        - `name`: output stem for this dataset
        - `path`: input relative to data.top_folder (or absolute): a `.root` file, or LUND input
          (see below)
        - `n_events`: default number of events to evaluate for this dataset (with `scatE_filter`,
          events the filter keeps)
- `event_model`
	Mapping with:
	- `ckpt_path`: checkpoint path
	- `sampler_type`: sampling algorithm name
	- `n_steps`: number of sampling steps
- `particle_model`
	Same structure as `event_model`.
- `impact_model` (optional)
	Same structure as `event_model`. If omitted, no impact model is used.
- `scatE_filter` (optional)
	Scattered-electron pre-filter, applied to every dataset while it is read, so only the kept
	events are loaded. Mapping with:
	- `ckpt_path`: classifier checkpoint (written by `train_scatE.py`)
	- `mode`: `sample` (default; keep each event with the classifier probability, which
	  reproduces the reconstruction efficiency) or `threshold`
	- `threshold`: probability cut of `threshold` mode (default 0.5)
	- `seed`: seed of `sample` mode (default 0); each dataset is sampled from a fresh generator
	Events without a truth electron are dropped. When the input carries the `pflow_scatE` label,
	the run prints a closure report against it. The settings and counts are written to the output
	as `metadata/scatE_filter`. Leave the block out for input that is already filtered.

Input formats:
- `.root`: the flat `evt_tree` of the training samples.
- LUND (`.dat`/`.lund`; a file, a directory of them, or a glob, read in name order): generator
  output. The 14-column particle lines become the `truth` collection: final-state particles
  (`type == 1`) other than electron neutrinos, with theta of at least 1 degree;
  `p = |(px, py, pz)|`, `theta = atan2(sqrt(px^2 + py^2), pz)`, `phi = atan2(py, px)`,
  `beta = p / sqrt(mass^2 + p^2)` from the file's mass column, vertices in cm as given.
  `eventNumber` counts the events of the input from 0. LUND input has no `pflow` collection.

Notes:
- `eval_path` can be relative; it is resolved by current working directory.
- `ckpt_path` files must exist.

## 2) Variable Config Structure

Reference example:
- [src/parnassus_core/evaluation/example_configs/eval_vars.yaml](./example_configs/eval_vars.yaml)

Fields and meaning:
- `event_vars`: Event-level branches read from the file, for example eventNumber.
- `vars_to_load`: Collection-wise variables loaded by dataset backend. `p`, `theta` and `phi`
	are required: the derived event variables and the particle ordering are built from them.
	Example:
	- `vars_to_load.truth: [p, theta, phi, vx, vy, vz, pdgId]`
- `fs_vars`
	Target variables generated by the model.
	Required keys:
	- `fs_vars.event`
	- `fs_vars.particle`
	Optional key:
	- `fs_vars.impact`
- `ctxt_vars`: Per-particle context inputs passed to the model. Should be the same as used during training.
- `ctxt_global_vars`: Event-level context vector entries. Should be the same as used during training.

Naming conventions:
- `truth_` and `pflow_` prefixes identify collection.

## 3) Running eval.py

Command template:
```bash
python -m parnassus_core.evaluation.eval \
    -c /path/to/eval_config.yaml \
    -bs 1024 \
    -ne 100000 \
    --suffix _v1
```

Arguments:
- `-c, --config` (required)
	Path to evaluation config YAML (analysis.yaml-style).
- `-bs, --batch_size`
	Dataloader batch size for eval mode.
- `-ne, --n_events`
	Override events per dataset. If omitted, each dataset uses data.datasets[].n_events.
- `--suffix`
	Extra string appended to the output filename stem.

What eval.py does:
- loads config and all requested checkpoints
- iterates over every dataset in data.datasets, reading it with the reader its path selects and,
  with `scatE_filter`, keeping only the selected events
- runs generation in eval mode
- writes ROOT files to eval_path

Output naming pattern: `<dataset_name>_eval_<k>k<suffix>.root`, where `k` is integer `n_events // 1000`.

## 4) Running postprocess_eval.py

Command template:
```bash
python -m parnassus_core.evaluation.postprocess_eval --ip -m -n -1 --eta 2.5 \
    -i /path/to/eval_file.root \
    -d /path/to/data_file.root \
    -o /path/to/output_file.root
```

Arguments:
- `-i, --input`
	Eval ROOT file produced by eval.py (fastsim branch source).
- `-d, --data` (required)
	Reference truth/pflow data ROOT file from the test dataset.
- `-o, --output`
	Output base filename. Postprocessor appends a suffix with event count and eta cut.
- `-n, --n_events`
	Number of events to process.
	Use -1 to process all matched events between input and data files.
- `--eta`
	Absolute eta acceptance cut applied to pflow and fastsim particles before jet clustering.
- `--ip`
	Include impact parameter variables in output processing.
- `-m, --matching`
	Enable particle matching metrics for all, jet-only, and background-only categories.
- `--dr_cut`
	Delta-R threshold for matching.
- `-dr`
	Jet clustering radius (anti-kt radius).
- `-df`
	Treat input as diffusion-style file (fastsim branch naming).
- `-dl`
	Treat input as Delphes-style file (different mask handling).

Output naming pattern: `<output_stem>_<n_events//1000>k_test_eta<eta*10>Eval.root`
