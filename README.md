# Parnassus for CLAS12 RG-A

Parnassus is a particle-level fast simulation based on flow matching. This repository applies it
to the CLAS12 detector at Jefferson Lab, Hall B, for the RG-A Fall 2018 inbending configuration:
a 10.6041 GeV electron beam on a proton target, trained on `clasdis` deep-inelastic-scattering
Monte Carlo. It holds the code for "An AI-based Detector Simulation and Reconstruction Model for the CLAS12 Detector at JLab" (arXiv: TODO).

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

- **This repository** is the release: the library, the CLAS12 configs, the scatE classifier,
  the job scripts, and the notebooks behind the paper figures.
- **Data preparation** (CLAS12 HIPO files to the ROOT trees used here) is in
  [parnassus_data_prep](https://github.com/tjayhllstrn/parnassus_data_prep); the tag
  `paper-v1` goes with the paper.
- **Trained checkpoints** of the paper models are in
  [`clas12_RGA/pre-trained-models/`](src/parnassus_core/experiments/clas12_RGA/pre-trained-models/README.md).

## Installation

Python 3.11 or newer:

```bash
git clone https://github.com/omphalos-mc/parnassus-clas12.git
cd parnassus-clas12
pip install -e . "jsonargparse[signatures]>=4.27.7"
```

This will cover all of the requirements, except the specific flash attention module, which needs to be built from your specific python, torch, and cuda version, as well as the CPU architecture used by your machine. After finding thes numbers, choose the pre-built wheel from this repository using the useful search feature:

https://github.com/mjun0812/flash-attention-prebuild-wheels/

and install with 

```bash
pip install <link_to_pre-build-wheel_with_python_torch_and_cuda_versions>
```

For convenience, run these commands in your environment to determine the relevant numbers for you.
```bash
uname -m
python --version
python -c "import torch; print('Torch version: ',torch.__version__); print('CUDA version: ',torch.version.cuda)"
```

### Installation on ifarm computing cluster

After cloning the repository, create an environment for the requirements. You may need to unset the PYTHONPATH environment variable to avoid attempting to install or uninstall from the CVMFS directory. 
For example:
```bash
conda create -n parnassus-env python=3.11
conda activate parnassus-env
conda env config vars set -n parnassus-env PYTHONPATH=""
pip install -e . "jsonargparse[signatures]>=4.27.7"
```

Then the pre-built wheel needed for ifarm at the time of this publication is the Linux x86_64, python 3.11, torch 2.9.1, CUDA 12.8 flash attention module. This can be installed with the following pip command:
```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl
```

## Quick start

Go to the CLAS12 README at src/parnassus_core/experiments/clas12_RGA/README.md for instructions on how to specify the input data set and run evaluation and/or training.

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

## Development

```bash
pip install pytest pytest-cov ruff pre-commit
pre-commit install
ruff check . && ruff format --check . && pytest
```

## Citation and license

If you use this code, please cite the paper; [`CITATION.cff`](CITATION.cff) has the reference.
The project is licensed under the MIT License; see [LICENSE.md](LICENSE.md).
