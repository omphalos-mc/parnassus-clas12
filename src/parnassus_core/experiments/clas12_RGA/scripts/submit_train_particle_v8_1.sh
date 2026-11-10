#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --job-name=clas12_particle_v8_1
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@90          # Lightning checkpoints + auto-requeues on this signal
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -eo pipefail

# --- Args ----------------------------------------------------------------
# Usage:
#   sbatch src/parnassus_core/experiments/clas12_RGA/scripts/submit_train_particle_v8_1.sh [--resume]   # per the #SBATCH block
#
#   salloc -A <account> -C gpu -q interactive -t 4:00:00 -N 4
#   bash src/parnassus_core/experiments/clas12_RGA/scripts/submit_train_particle_v8_1.sh [--resume]     # adapts to the allocation
#
#   --resume : on a FRESH sbatch (SLURM_RESTART_COUNT=0), resume from the newest
#              checkpoint for this run instead of starting over. (A Slurm requeue
#              after a node failure already auto-resumes without this flag; hitting
#              the wall-clock limit does not auto-requeue, so use --resume then.)
RESUME=0
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --resume) RESUME=1 ;;
        --dry_run) DRY_RUN=1 ;;
        *) echo "Unknown argument: ${arg}" >&2
           echo "Usage: sbatch $0 [--resume]" >&2
           exit 1 ;;
    esac
done

# --- Environment ---------------------------------------------------------
# Submit from the repository root with the parnassus-core environment active;
# sbatch passes that environment on to the job. DATA is the directory holding
# v8/v8_train.root. The online CometLogger reads COMET_API_KEY and
# COMET_WORKSPACE from the environment; without them, run the logger offline.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA:?export DATA=<directory holding v8/v8_train.root>}"
DATA_ARGS=(--data.train_path "${DATA}/v8/v8_train.root" --data.valid_path "${DATA}/v8/v8_train.root")

# Avoid CPU thread oversubscription across the DDP ranks.
export OMP_NUM_THREADS=8 #"${SLURM_CPUS_PER_TASK:-14}"

# Our checkpoints are trusted (self-produced) but contain custom pickled classes
# (Lion, FlowMatcher, ...) that PyTorch >=2.6's default weights_only=True load rejects.
# Force weights_only=False for the resume load. (This covers the trainer connector, which
# loads with weights_only unset; the LightningCLI pre-parse is handled in main.py.)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# name experiment after the job ID so the same logger can be picked up if there is a requeue.
# COMET_EXP_ID is the name cli.py reads and passes as the CometLogger experiment_key
# (matches the legacy submit_wis.py convention). A pre-set COMET_EXP_ID is respected, so
#   COMET_EXP_ID=<old key> sbatch src/parnassus_core/experiments/clas12_RGA/scripts/submit_train_particle_v8_1.sh --resume
# continues the SAME Comet page; otherwise a fresh sbatch (new job id) gets its own page.
export COMET_EXP_ID="${COMET_EXP_ID:-$(printf 'clas12pid%023d' "${SLURM_JOB_ID}")}"

cd "${REPO_ROOT}"

# --- Resource geometry ----------------------------------------------------
# The #SBATCH block above only applies under sbatch. Run inside an salloc shell
# it is inert, and srun then inherits no GPU request (SLURM_GPUS_PER_NODE is
# unset), so every task lands in a cgroup with zero visible GPUs and Lightning
# dies with "No supported gpu backend found!". Passing the resources to srun
# explicitly makes the sbatch and salloc paths identical.
# (--dry_run runs plain python with no srun, so it is fine on a login node.)
if [[ -z "${SLURM_JOB_ID:-}" && "${DRY_RUN}" -eq 0 ]]; then
    echo "ERROR: no Slurm allocation found. Use sbatch, or run inside salloc:" >&2
    echo "  salloc -A <account> -C gpu -q interactive -t 4:00:00 -N 4" >&2
    exit 1
fi

NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
CPUS_PER_TASK=32

CONFIG=src/parnassus_core/experiments/clas12_RGA/configs/v8/particle_v8_1.yaml
NAME=v8_1_clas12_RGA_particle   # must match `name:` in particle_v8_1.yaml

# --- Auto-resume: on requeue, or on a fresh sbatch with --resume ---------
# This partition isn't preemptible, so the only automatic Slurm requeue here is
# on node failure (--requeue above), which restarts the batch script from the
# top (not mid-execution); SigtermCheckpoint / ModelCheckpoint write the
# checkpoints and cli.py re-timestamps the output dir, so we pick up the newest
# checkpoint for this run. Hitting the wall-clock limit does NOT auto-requeue --
# SIGUSR1@90 below just gives Lightning time to checkpoint before the timeout
# kill, so a follow-up run needs --resume to pick it up. A fresh sbatch (count 0)
# starts clean UNLESS --resume is passed, which picks up the newest
# last.ckpt/sigterm.ckpt from a prior run of this NAME.
RESUME_ARGS=()
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 || "${RESUME}" -eq 1 ]]; then
    LAST_CKPT=$(find "${REPO_ROOT}/logs" -maxdepth 6 -type f \
        \( -name 'last.ckpt' -o -name 'sigterm.ckpt' \) \
        -path "*${NAME}_*" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -n1 | cut -d' ' -f2- || true)
    if [[ -n "${LAST_CKPT}" ]]; then
        echo "Resuming from ${LAST_CKPT}  (restart=${SLURM_RESTART_COUNT:-0}, --resume=${RESUME})"
        RESUME_ARGS=(--ckpt_path "${LAST_CKPT}")
    else
        echo "Resume requested (restart=${SLURM_RESTART_COUNT:-0}, --resume=${RESUME}) but no checkpoint found for ${NAME}; starting fresh"
    fi
fi

# --- Info ----------------------------------------------------------------
echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Nodes:     ${NNODES}  Tasks/node: ${GPUS_PER_NODE}  GPUs/node: ${GPUS_PER_NODE}"
echo "Config:    ${CONFIG}"
echo "Restarts:  ${SLURM_RESTART_COUNT:-0}"
echo "Started:   $(date)"

# --- Run -----------------------------------------------------------------
# base.yaml is auto-merged as the fit default config (see main.py); the
# stage config is layered on top with -c. srun launches one process per
# task so Lightning's SLURM DDP integration wires up the single rank.
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run: running fast dev run"
    python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c "${CONFIG}" \
    --matmul_precision high \
    --trainer.fast_dev_run=1 \
    --trainer.devices=1 \
    --trainer.num_nodes=1 \
    --trainer.strategy auto \
    "${DATA_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
    exit 0
fi

SRUN_ARGS=(
    --nodes="${NNODES}"
    --ntasks-per-node="${GPUS_PER_NODE}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --gpu-bind=none
)

# Preflight: one cheap step that proves every rank can see its GPUs, so a
# misconfigured allocation fails in seconds instead of after dataset loading.
srun "${SRUN_ARGS[@]}" bash -c \
    'echo "rank ${SLURM_PROCID} $(hostname): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"'

srun "${SRUN_ARGS[@]}" python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c "${CONFIG}" \
    --matmul_precision high \
    --trainer.num_nodes="${NNODES}" \
    --trainer.devices="${GPUS_PER_NODE}" \
    "${DATA_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

echo "Finished:  $(date)"
