#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --job-name=clasdis_v8_1_eval_chain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Run the full CLAS12 v8.1 evaluation chain end to end, with the pre-trained scatE
# classifier supplying the scattered-electron reconstruction efficiency:
#
#   1. scatE_filter  --mode sample  : AllEvents -> classifier-filtered fastsim input
#   2. scatE_filter  --mode truth   : AllEvents -> reference for step 4's -d
#   3. eval                         : event -> particle full-chain generation
#   4. postprocess_eval             : cuts, matching, DIS kinematics, four-vectors
#
# Steps 1 and 2 are skipped automatically if their output files already exist,
# so a re-run costs only steps 3-4. Use --refilter to force them.
#
# Why two filter passes: the classifier's keep decision is stochastic and
# independent of what reconstruction actually did, so ~21.5% of Bernoulli-kept
# events genuinely have pflow_scatE == 0 and their pflow_* branches are not a
# valid reference. postprocess_eval joins -i to -d by eventNumber, so pointing -d
# at the truth-mode file restricts the comparison to events that were both
# classifier-kept and truly reconstructed. See the "scatE pre-filter" section of
# src/parnassus_core/experiments/clas12_RGA/README.md.
#
# Usage:
#   sbatch src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain_v8_1.sh [options]
#
#   The paper run (every classifier-kept event of all 29.91M in v8_test.root; postprocess_eval
#   was then run by hand with -n 6000000, see scripts/README.md):
#   sbatch src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain_v8_1.sh --refilter --n-read 29910000 --n-keep 29910000 --suffix v8_filter_29M_test --skip-postprocess
#
#     --refilter          Rebuild the filtered inputs even if they already exist.
#     --n-keep N          Target kept events for the sampled input (default 500000).
#     --n-read N          Input events to scan (default 2700000, ~5.4x --n-keep; -1 reads all).
#     --n-events N        Events to evaluate (default: --n-keep).
#     -bs, --batch-size N Eval dataloader batch size (default 64).
#     --seed N            Bernoulli seed for the sampled filter (default 0).
#     --suffix S          Suffix for the eval output filename stem.
#     --skip-postprocess  Stop after eval.py.
#     -h, --help          Show this help.

set -eo pipefail

# --- Args ----------------------------------------------------------------
REFILTER=0
N_KEEP=500_000
N_READ=2_700_000
N_EVENTS=""
BATCH_SIZE=4096
SEED=0
SUFFIX=""
SKIP_POSTPROCESS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --refilter)         REFILTER=1; shift ;;
        --n-keep)           N_KEEP="$2"; shift 2 ;;
        --n-read)           N_READ="$2"; shift 2 ;;
        --n-events)         N_EVENTS="$2"; shift 2 ;;
        -bs|--batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --suffix)           SUFFIX="$2"; shift 2 ;;
        --skip-postprocess) SKIP_POSTPROCESS=1; shift ;;
        -h|--help)          grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2
           echo "Usage: sbatch $0 [--refilter] [--n-keep N] [--n-read N] [--n-events N] [-bs N] [--seed N] [--suffix S] [--skip-postprocess]" >&2
           exit 1 ;;
    esac
done
: "${N_EVENTS:=${N_KEEP}}"

# --- Environment ---------------------------------------------------------
# Submit from the repository root with the parnassus-core environment active;
# sbatch passes that environment on to the job. DATA is the directory holding
# v8/v8_test.root, i.e. data.top_folder of EVAL_CONFIG.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA:?export DATA=<directory holding v8/v8_test.root>}"

# FlowLightning checkpoints are loaded with weights_only=False.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "${REPO_ROOT}"

# --- Paths ---------------------------------------------------------------
FILTER=parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter
CONFIG_DIR=src/parnassus_core/experiments/clas12_RGA/evaluation/configs/v8
EVAL_CONFIG="${CONFIG_DIR}/config_v8_1.yaml"
POSTPROCESS_CONFIG="${CONFIG_DIR}/postprocess_union.yaml"

# Must agree with EVAL_CONFIG's data.top_folder + datasets[0].path, since
# DatasetConfig asserts the path exists.
SOURCE_ROOT="${DATA}/v8/v8_test.root"
SAMPLE_ROOT="${DATA}/v8/v8_test_filtered.root"
TRUTH_ROOT="${DATA}/v8/v8_test_filtered_truth.root"

# Must agree with EVAL_CONFIG's eval_path + datasets[0].name.
EVAL_DIR="clasdis_v8_1_filtered"
DATASET_NAME="clasdis_v8_filtered"

mkdir -p "${REPO_ROOT}/logs/slurm"

# --- Info ----------------------------------------------------------------
echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "Source:      ${SOURCE_ROOT}"
echo "Eval config: ${EVAL_CONFIG}"
echo "n_read/n_keep/n_events: ${N_READ} / ${N_KEEP} / ${N_EVENTS}   batch=${BATCH_SIZE} seed=${SEED}"
echo "Started:     $(date)"

# --- 1. Filter: sampled fastsim input ------------------------------------
if [[ ! -f "${SAMPLE_ROOT}" || "${REFILTER}" -eq 1 ]]; then
    echo -e "\n=== [1/4] scatE filter (--mode sample) -> ${SAMPLE_ROOT} ==="
    python -m "${FILTER}" \
        -i "${SOURCE_ROOT}" \
        -o "${SAMPLE_ROOT}" \
        --mode sample \
        -ne "${N_READ}" \
        --n-keep "${N_KEEP}" \
        --seed "${SEED}"
else
    echo -e "\n=== [1/4] reusing existing ${SAMPLE_ROOT} (pass --refilter to rebuild) ==="
fi

echo "time:    $(date)"

# --- 2. Filter: truth reference ------------------------------------------
if [[ ! -f "${TRUTH_ROOT}" || "${REFILTER}" -eq 1 ]]; then
    echo -e "\n=== [2/4] scatE filter (--mode truth) -> ${TRUTH_ROOT} ==="
    # Same read window as step 1 so the eventNumber join in step 4 has full
    # coverage of the sampled events.
    python -m "${FILTER}" \
        -i "${SOURCE_ROOT}" \
        -o "${TRUTH_ROOT}" \
        --mode truth \
        -ne "${N_READ}"
else
    echo -e "\n=== [2/4] reusing existing ${TRUTH_ROOT} (pass --refilter to rebuild) ==="
fi

echo "time:    $(date)"

# --- 3. Full-chain evaluation --------------------------------------------
echo -e "\n=== [3/4] eval.py (event -> particle) ==="
SUFFIX_ARGS=()
[[ -n "${SUFFIX}" ]] && SUFFIX_ARGS=(--suffix "${SUFFIX}")
python -m parnassus_core.evaluation.eval \
    -c "${EVAL_CONFIG}" \
    -bs "${BATCH_SIZE}" \
    -ne "${N_EVENTS}" \
    --num_workers 8 \
    "${SUFFIX_ARGS[@]}"

# eval.py names its output <name>_eval_<n//1000>k<suffix>.root, where n is the
# count *after* generate_event's event-level veto, so it is not known ahead of
# time -- resolve it as the newest matching file rather than reconstructing it.
EVAL_OUTPUT=$(find "${EVAL_DIR}" -maxdepth 1 -type f \
    -name "${DATASET_NAME}_eval_*k${SUFFIX}.root" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -n1 | cut -d' ' -f2- || true)
if [[ -z "${EVAL_OUTPUT}" ]]; then
    echo "ERROR: no eval output found under ${EVAL_DIR}" >&2
    exit 1
fi
echo "Eval output: ${EVAL_OUTPUT}"

echo "time:    $(date)"

# --- 4. Postprocessing ---------------------------------------------------
if [[ "${SKIP_POSTPROCESS}" -eq 1 ]]; then
    echo -e "\n=== [4/4] skipped (--skip-postprocess) ==="
else
    echo -e "\n=== [4/4] postprocess_eval.py ==="
    # -d is the TRUTH-mode file, never SAMPLE_ROOT: see the header comment.
    python -m parnassus_core.evaluation.postprocess_eval \
        -i "${EVAL_OUTPUT}" \
        -d "${TRUTH_ROOT}" \
        -o "${EVAL_DIR}/postprocess${SUFFIX}.root" \
        -n -1 \
        -c "${POSTPROCESS_CONFIG}"
fi

echo -e "\nFinished:    $(date)"
