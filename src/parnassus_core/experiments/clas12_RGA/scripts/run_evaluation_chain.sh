#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --job-name=clasdis_eval_chain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Run the full CLAS12 evaluation chain end to end, with the pre-trained scatE
# classifier supplying the scattered-electron reconstruction efficiency:
#
#   1. scatE_filter        : AllEvents -> pflow_scatE == 1 reference for step 3's -d
#   2. eval                : event -> particle full-chain generation; the
#                            scatE_filter block of the eval config keeps each
#                            event with the classifier's probability while the
#                            input is read
#   3. postprocess_eval    : cuts, matching, DIS kinematics, four-vectors
#
# Step 1 is skipped automatically if its output file already exists, so a re-run
# costs only steps 2-3. Use --refilter to force it.
#
# Why a separate reference: the classifier's keep decision is stochastic and
# independent of what reconstruction actually did, so ~21.5% of Bernoulli-kept
# events genuinely have pflow_scatE == 0 and their pflow_* branches are not a
# valid reference. postprocess_eval joins -i to -d by eventNumber, so pointing -d
# at the pflow_scatE == 1 file restricts the comparison to events that were both
# classifier-kept and truly reconstructed. See the "scatE pre-filter" section of
# src/parnassus_core/experiments/clas12_RGA/README.md.
#
# Usage:
#   sbatch src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain.sh [options]
#
#   The paper run (every classifier-kept event of all 29.91M in v8_test.root; postprocess_eval
#   was then run by hand with -n 6000000, see scripts/README.md):
#   sbatch src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain.sh --refilter --n-read 29910000 --n-events 29910000 --suffix _paper --skip-postprocess
#
#     --refilter          Rebuild the reference even if it already exists.
#     --n-events N        Classifier-kept events to evaluate (default 500000).
#     --n-read N          Input events the reference covers (default 2700000, ~5.4x
#                         --n-events; -1 reads all). Must cover every event eval scans.
#     -bs, --batch-size N Eval dataloader batch size (default 4096).
#     --suffix S          Suffix for the eval output filename stem.
#     --skip-postprocess  Stop after eval.py.
#     -h, --help          Show this help.
#
#   The classifier checkpoint, mode and seed come from the scatE_filter block of
#   evaluation/configs/config.yaml.

set -eo pipefail

# --- Args ----------------------------------------------------------------
REFILTER=0
N_EVENTS=500_000
N_READ=2_700_000
BATCH_SIZE=4096
SUFFIX=""
SKIP_POSTPROCESS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --refilter)         REFILTER=1; shift ;;
        --n-events)         N_EVENTS="$2"; shift 2 ;;
        --n-read)           N_READ="$2"; shift 2 ;;
        -bs|--batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --suffix)           SUFFIX="$2"; shift 2 ;;
        --skip-postprocess) SKIP_POSTPROCESS=1; shift ;;
        -h|--help)          grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2
           echo "Usage: sbatch $0 [--refilter] [--n-events N] [--n-read N] [-bs N] [--suffix S] [--skip-postprocess]" >&2
           exit 1 ;;
    esac
done

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
CONFIG_DIR=src/parnassus_core/experiments/clas12_RGA/evaluation/configs
EVAL_CONFIG="${CONFIG_DIR}/config.yaml"
POSTPROCESS_CONFIG="${CONFIG_DIR}/postprocess_union.yaml"

# Must agree with EVAL_CONFIG's data.top_folder + datasets[0].path, the input
# eval.py reads and filters.
SOURCE_ROOT="${DATA}/v8/v8_test.root"
TRUTH_ROOT="${DATA}/v8/v8_test_filtered_truth.root"

# Must agree with EVAL_CONFIG's eval_path + datasets[0].name.
EVAL_DIR="clasdis_filtered"
DATASET_NAME="clasdis_filtered"

mkdir -p "${REPO_ROOT}/logs/slurm"

# --- Info ----------------------------------------------------------------
echo "Host:        $(hostname)"
echo "Job:         ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "Source:      ${SOURCE_ROOT}"
echo "Eval config: ${EVAL_CONFIG}"
echo "n_events/n_read: ${N_EVENTS} / ${N_READ}   batch=${BATCH_SIZE}"
echo "Started:     $(date)"

# --- 1. Reference: pflow_scatE == 1 events -------------------------------
if [[ ! -f "${TRUTH_ROOT}" || "${REFILTER}" -eq 1 ]]; then
    echo -e "\n=== [1/3] scatE_filter (pflow_scatE == 1) -> ${TRUTH_ROOT} ==="
    # The read window must cover every event the eval run's classifier pass
    # scans, so the eventNumber join in step 3 has full coverage.
    python -m "${FILTER}" \
        -i "${SOURCE_ROOT}" \
        -o "${TRUTH_ROOT}" \
        -ne "${N_READ}"
else
    echo -e "\n=== [1/3] reusing existing ${TRUTH_ROOT} (pass --refilter to rebuild) ==="
fi

echo "time:    $(date)"

# --- 2. Full-chain evaluation --------------------------------------------
echo -e "\n=== [2/3] eval.py (event -> particle, scatE-filtered input) ==="
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

# --- 3. Postprocessing ---------------------------------------------------
if [[ "${SKIP_POSTPROCESS}" -eq 1 ]]; then
    echo -e "\n=== [3/3] skipped (--skip-postprocess) ==="
else
    echo -e "\n=== [3/3] postprocess_eval.py ==="
    # -d is the pflow_scatE == 1 reference: see the header comment.
    python -m parnassus_core.evaluation.postprocess_eval \
        -i "${EVAL_OUTPUT}" \
        -d "${TRUTH_ROOT}" \
        -o "${EVAL_DIR}/postprocess${SUFFIX}.root" \
        -n -1 \
        -c "${POSTPROCESS_CONFIG}"
fi

echo -e "\nFinished:    $(date)"
