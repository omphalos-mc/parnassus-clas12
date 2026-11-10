# Parnassus for CLAS12 RG-A

Parnassus is a particle-level fast simulation based on flow matching. This repository applies it
to the CLAS12 detector at Jefferson Lab, Hall B, for the RG-A Fall 2018 inbending configuration:
a 10.6041 GeV electron beam on a proton target, trained on `clasdis` deep-inelastic-scattering
Monte Carlo. It holds the code for "TODO: paper title" (arXiv: TODO).

**Start with the [CLAS12 README](src/parnassus_core/experiments/clas12_RGA/README.md)**. It
covers the generation chain, the data, training, evaluation and the paper figures. The rest of
the package is the shared Parnassus core library.

## The fast simulation

Each event goes through three stages:

1. **scatE pre-filter.** A small classifier
   ([`scatE_classifier/`](src/parnassus_core/experiments/clas12_RGA/scatE_classifier/README.md))
   gives the probability that reconstruction finds the scattered electron, and each event is
   kept by a Bernoulli draw on that probability.
2. **Event model.** Generates the reconstructed multiplicity, the event-level momentum sums and
   the photon and neutron counts, conditioned on the truth particles.
3. **Particle model.** Generates every reconstructed particle's momentum, angles, velocity,
   vertex and `pdgId`, conditioned on the truth particles and the event-model output.

Postprocessing then applies the acceptance cuts, matches the generated particles to the truth
particles, computes the DIS kinematics and joins the fast-sim output with the full-sim reference.

## What is in this repository

- **`main`** (this branch) is the release: the library, the CLAS12 configs, the scatE
  classifier, the job scripts, and the notebooks behind the paper figures.
- **`train_clas12`** is the informal development branch, with the analysis tools in `tools/`,
  the configs of the earlier models, and the full history. The `pre-cleanup` tag can rerun
  those models.
- **Data preparation** (CLAS12 HIPO files to the ROOT trees used here) is in
  [parnassus_data_prep](https://github.com/tjayhllstrn/parnassus_data_prep).
- **Trained checkpoints and the postprocessed MC evaluation output** are on Zenodo
  (DOI: TODO). Real CLAS12 RG-A data are not distributed.

## Installation

Python 3.11 or newer:

```bash
git clone --single-branch -b main https://github.com/tjayhllstrn/parnassus-core-clas12-configs.git
cd parnassus-core-clas12-configs
pip install -e . "jsonargparse[signatures]>=4.27.7"
```

## Quick start

Run from the repository root. The data and checkpoint paths in the configs are site-specific;
the CLAS12 README says which ones to edit.

```bash
# train the event and particle models
python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c src/parnassus_core/experiments/clas12_RGA/configs/event.yaml
python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c src/parnassus_core/experiments/clas12_RGA/configs/particle.yaml

# full chain on a 500k-event sample: pre-filter -> generation -> postprocessing
export DATA=<directory holding v8/v8_test.root>
sbatch -A <account> src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain.sh
```

The [job scripts README](src/parnassus_core/experiments/clas12_RGA/scripts/README.md) gives the
exact commands of the paper run, and the notebooks in
[`clas12_RGA/notebooks/`](src/parnassus_core/experiments/clas12_RGA/notebooks) make the paper
figures from its output.

## The core library

The CLAS12 experiment uses these parts of the library:

```text
src/parnassus_core/
├── data/           # EagerDataset (ROOT trees into memory), ParnassusDataModule, feature transforms
├── evaluation/     # eval.py (full-chain generation) and postprocess_eval.py
├── experiments/    # clas12_RGA/, plus the CMS experiments described below
├── flow/           # FlowLightning model wrapper, networks, losses, samplers, callbacks
├── performance/    # shared performance and plotting utilities
├── preprocessing/  # scaling/calculate_transforms_batched.py, the feature-scaling statistics
└── utils/          # shared helpers, including the pdgId one-hot encoding (utils/pid.py)
```

Both models are trained with the shared `FlowLightning` wrapper and the Lightning CLI, and read
the ROOT trees through `ParnassusDataModule`. The [data guide](src/parnassus_core/data/README.md),
the [evaluation guide](src/parnassus_core/evaluation/README.md) and the
[scaling guide](src/parnassus_core/preprocessing/scaling/README.md) describe the configuration
options.

## Other experiments

The core library also includes the CMS 2011 and CMS 2012 experiments, which are not part of the
CLAS12 work. They add an impact-parameter model and their own preprocessing pipeline; see the
[CMS 2011](src/parnassus_core/experiments/cms_2011/README.md) and
[CMS 2012](src/parnassus_core/experiments/cms_2012/README.md) READMEs and the
[preprocessing guide](src/parnassus_core/preprocessing/README.md).

## Development

```bash
pip install pytest pytest-cov ruff pre-commit
pre-commit install
ruff check . && ruff format --check . && pytest
```

## Citation and license

If you use this code, please cite the paper; [`CITATION.cff`](CITATION.cff) has the reference.
The project is licensed under the MIT License; see [LICENSE.md](LICENSE.md).
