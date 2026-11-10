#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --job-name=clasdis_speed_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Time every stage of the CLAS12 inference chain on 500k generated events:
# scatE pre-filter, dataset load, batch assembly, event network, particle
# network, unscaling, and output writing. The report is printed and saved to
# clasdis_filtered/logs/speed_test_<timestamp>.{log,json}.
#
# On an interactive GPU node:
#   salloc -A <account> -C gpu -q shared_interactive -G 1 -c 32 -t 03:00:00
#   bash src/parnassus_core/experiments/clas12_RGA/scripts/run_speed_test.sh [speed_test options]
#
# Or as a batch job: sbatch src/parnassus_core/experiments/clas12_RGA/scripts/run_speed_test.sh [speed_test options]
#
# Extra arguments are forwarded to speed_test.py (and override the defaults
# below), e.g.
#   -bs N              Generation batch size (default 4096, as in the eval chain).
#   --num-workers N    DataLoader workers (default 8 here; 0 = in-process).
#   -ne N              Events to generate (default 500000, from the config).
#   --skip-write       Skip the final eval ROOT write.
#   --keep-outputs     Keep the transient filtered/eval ROOT files in $SCRATCH.
# See `python -m parnassus_core.experiments.clas12_RGA.evaluation.speed_test -h`.

set -eo pipefail

# Submit from the repository root with the parnassus-core environment active;
# sbatch passes that environment on to the job. DATA is the directory holding
# v8/v8_test.root, i.e. data.top_folder of EVAL_CONFIG.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA:?export DATA=<directory holding v8/v8_test.root>}"

# FlowLightning checkpoints are loaded with weights_only=False.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# eval_path and the classifier checkpoint are relative to the repo root.
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"

python -m parnassus_core.experiments.clas12_RGA.evaluation.speed_test \
    -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/config.yaml \
    --filter-input "${DATA}/v8/v8_test.root" \
    --num-workers 8 \
    "$@"
