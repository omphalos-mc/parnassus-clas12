# Parnassus Core

Core training, preprocessing, inference, and evaluation code for Parnassus, a
particle-level fast-simulation framework based on flow matching.

The package provides shared model and data infrastructure together with
experiment configurations for CMS 2011 and CMS 2012 data.

## Installation

Parnassus Core requires Python 3.11 or newer.

```bash
git clone https://github.com/parnassus-hep/parnassus-core.git
cd parnassus-core
pip install -e . "jsonargparse[signatures]>=4.27.7"
```

Development tools are listed in the `dev` dependency group in
`pyproject.toml`. They can also be installed directly:

```bash
pip install pytest pytest-cov ruff pre-commit
pre-commit install
```

Run the checks with:

```bash
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

## Experiments

Experiment implementations live under `src/parnassus_core/experiments/`.
Each experiment uses the shared `FlowLightning` model wrapper,
`ParnassusDataModule`, and Lightning CLI. Experiment directories provide the
training entry point, model/data configurations, scaling parameters, and
experiment-specific performance code.

Available experiments:

- [CMS 2011](src/parnassus_core/experiments/cms_2011/README.md)
- [CMS 2012](src/parnassus_core/experiments/cms_2012/README.md)

Both experiments implement a three-stage generation chain:

1. **Event model**: event-level quantities such as multiplicity, missing
   transverse momentum, and scalar transverse momentum.
2. **Particle model**: particle kinematics, class, charge, and vertex
   quantities conditioned on event-level information.
3. **Impact model**: track impact parameters and uncertainties conditioned on
   generated particles.

CMS 2012 uses `vz_shifted = vz - pv_z` for longitudinal vertex modeling.

### Experiment Layout

```text
experiments/
├── cms_2011/
│   ├── main.py
│   ├── configs/
│   ├── performance/
│   ├── export/
│   ├── delphes/
│   └── notebooks/
└── cms_2012/
    ├── main.py
    ├── configs/
    ├── evaluation/
    └── performance/
```

The main configuration groups are:

- `base.yaml`: shared data paths, trainer settings, logging, and optimization
- `event.yaml`, `particle.yaml`, `impact.yaml`: stage-specific model settings
- `*_vars.yaml`: variables loaded as targets and conditioning
- `var_transform.yaml`: feature normalization and transform parameters

### Training

Run an experiment from the repository root:

```bash
# CMS 2012 examples
python -m parnassus_core.experiments.cms_2012.main fit \
  -c src/parnassus_core/experiments/cms_2012/configs/event.yaml

python -m parnassus_core.experiments.cms_2012.main fit \
  -c src/parnassus_core/experiments/cms_2012/configs/particle.yaml

python -m parnassus_core.experiments.cms_2012.main fit \
  -c src/parnassus_core/experiments/cms_2012/configs/impact.yaml
```

Replace `cms_2012` with `cms_2011` to run the corresponding 2011 experiment.
Paths and trainer settings in `configs/base.yaml` are site-specific and should
be updated before training.

For a short configuration and dataloader check:

```bash
python -m parnassus_core.experiments.cms_2012.main fit \
  -c src/parnassus_core/experiments/cms_2012/configs/event.yaml \
  --trainer.fast_dev_run
```

### Prediction

Use Lightning's `test` command with a trained checkpoint:

```bash
python -m parnassus_core.experiments.cms_2012.main test \
  -c src/parnassus_core/experiments/cms_2012/configs/event.yaml \
  --ckpt_path /path/to/checkpoint.ckpt \
  --data.test_path /path/to/test.h5 \
  --data.num_test 100000
```

See the experiment READMEs for stage-specific prediction behavior and output
naming.

## Preprocessing

Preprocessing tools live in `src/parnassus_core/preprocessing/`. The primary
CMS pipeline is:

```text
POET CMS ROOT
    -> cms_to_hdf5.py
    -> merge_shuffle_hdf5.py
    -> scaling/calculate_transforms_batched.py
```

Example direct conversion:

```bash
python -m parnassus_core.preprocessing.cms_to_hdf5 \
  --input /path/to/poet/root/files \
  --output /path/to/output.h5 \
  --n_events -1 \
  --pt_cut 0.25 \
  --pflow-pt-cut 1.0 \
  --max_particles '{"truth": 400, "pflow": 400, "hadrons": 50}'
```

Example sequential merge into multiple output files:

```bash
python -m parnassus_core.preprocessing.merge_shuffle_hdf5 \
  --input '/path/to/input/*.h5' \
  --output /path/to/merged.h5 \
  --no-shuffle \
  --events-per-file 1400000 \
  --skip-overlap-check
```

The preprocessing package also includes the older CMS ROOT-to-flat-ROOT
pipeline, Delphes and HepMC converters, scaling utilities, and a legacy PBS
submission helper. See the
[preprocessing guide](src/parnassus_core/preprocessing/README.md) for all
scripts, options, and data layouts.

## Data

`parnassus_core.data` provides a common physics dataset interface:

- `EagerDataset` loads ROOT data into memory.
- `StreamingDataset` reads padded HDF5 data in batches.
- `ParnassusDataModule` configures train, validation, and test loaders for
  Lightning.
- `RandomBatchSampler` provides shuffled batch slices for streaming datasets.

Both backends produce the same model-facing tensors and share feature
transforms, computed variables, masks, and context assembly. See the
[data guide](src/parnassus_core/data/README.md) for configuration and API
examples.

## Evaluation

The shared evaluation pipeline runs trained event, particle, and optional
impact models over one or more datasets:

```bash
python -m parnassus_core.evaluation.eval \
  --config src/parnassus_core/evaluation/example_configs/eval_config.yaml
```

Postprocess generated ROOT files to apply acceptance cuts, cluster jets, and
optionally calculate particle matching:

```bash
python -m parnassus_core.evaluation.postprocess_eval \
  --input /path/to/eval.root \
  --data /path/to/reference.root \
  --output /path/to/postprocessed.root \
  --n_events -1
```

See the [evaluation guide](src/parnassus_core/evaluation/README.md) for the
configuration schema and complete command-line options.

## Model Export

Trained models can be exported to ONNX or Torch format:

```bash
python -m parnassus_core.flow.export.export \
  --ckpt /path/to/checkpoint.ckpt \
  --config /path/to/config.yaml \
  --output /path/to/model.onnx \
  --mode onnx
```

## Package Structure

```text
src/parnassus_core/
├── data/           # ROOT/HDF5 datasets, loaders, transforms, data module
├── evaluation/     # Full-chain inference and ROOT postprocessing
├── experiments/    # CMS 2011 and CMS 2012 configs and entry points
├── flow/           # Models, modules, losses, samplers, callbacks, export
├── performance/    # Shared physics-performance and plotting utilities
├── preprocessing/  # ROOT/HDF5 conversion, merging, scaling, format bridges
└── utils/          # Shared utility functions
```

## License

The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
