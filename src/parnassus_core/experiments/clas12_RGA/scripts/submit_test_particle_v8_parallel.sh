#!/bin/bash
# Evaluate several particle-model checkpoints in parallel inside an interactive allocation,
# then postprocess each prediction file against the test data with
# evaluation/configs/v8/postprocess_df.yaml.
#
# Each checkpoint is tested once per solver, given as -s <sampler>:<n_steps> (sampler is one
# of euler, sde, pndm, dpm; default dpm:40). Each (checkpoint, solver) test runs as its own
# one-GPU slot: a test job step followed by a postprocess job step. With N tests and n
# allocated tasks ($SLURM_NTASKS), the first n start immediately and each remaining test
# starts as soon as a running slot finishes.
# Both outputs land in the run dir, named from
# <output_name> = <run_tag>_particleNetTest_<ckpt_tag>[_<EMA/SWA suffix>][_<sampler>]_s<n_steps>,
# where <run_tag> is the run dir name without its timestamp, <ckpt_tag> is epoch<NNN> for
# epoch=<NNN>-... checkpoints, or the checkpoint name (last, sigterm) otherwise, and
# _<sampler> is left out for dpm so dpm outputs keep their earlier _s<n_steps> names:
#   test:        <output_name>_500k.root
#   postprocess: <output_name>_postprocess_dr_0_3_500k_test_eta49Eval.root
#
# Checkpoints can also be picked from a run dir by epoch and weights, instead of or as well as
# listing them: -d <run_dir> -e 3,51,90 -w none,EMA tests the raw and -EMA checkpoints of
# epochs 3, 51 and 90. -w takes none, EMA or SWA (default none), and -d also accepts
# <run_dir>/ckpts. Each (epoch, weights) pair must match exactly one checkpoint.
#
# Allocate 1-4 nodes (4 GPUs each), then run:
#   salloc -A <account> -C gpu -q interactive -N 2 -t 04:00:00 \
#       --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32
#   bash src/parnassus_core/experiments/clas12_RGA/scripts/submit_test_particle_v8_parallel.sh [-s dpm:60 -s euler:100 ...] \
#       <run_dir>/ckpts/<a>.ckpt <run_dir>/ckpts/<b>.ckpt ...
#   bash src/parnassus_core/experiments/clas12_RGA/scripts/submit_test_particle_v8_parallel.sh [-s dpm:60 ...] -d <run_dir> -e 3,51,90 [-w EMA]
#
# Both steps of a test log to logs/slurm/<output_name>_<jobid>.log. A failed test
# skips its postprocess step. Failed evaluations do not stop the others; a summary is
# printed at the end and the exit code is non-zero if any failed. Press Ctrl-C twice
# within a second to cancel the running job steps.

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LOG_DIR="${REPO_ROOT}/logs/slurm"
# Events to test and postprocess; both outputs are named with N_TEST / 1000 (500k).
N_TEST=500000
# Solver used when no -s is given. Outputs carry _s<n_steps> (and the sampler unless it is
# dpm) so tests with different solvers do not overwrite each other.
DEFAULT_SOLVER=dpm:40
# Relative to the repo root. The _dr_0_3 in the postprocess output
# name is this config's dr_cut.
POSTPROCESS_CONFIG=src/parnassus_core/experiments/clas12_RGA/evaluation/configs/v8/postprocess_df.yaml

usage() {
    echo "Usage: $(basename "$0") [-s <sampler>:<n_steps>]..." \
        "[-d <run_dir> -e <epoch>[,<epoch>...] [-w none|EMA|SWA[,...]]]" \
        "[<run_dir>/ckpts/<checkpoint>.ckpt ...]" >&2
}

SOLVERS=()
CKPT_DIR=""
EPOCHS=()
WEIGHTS=()
while getopts "s:d:e:w:" opt; do
    case "${opt}" in
        s)
            if [[ ! ${OPTARG} =~ ^(euler|sde|pndm|dpm):[1-9][0-9]*$ ]]; then
                echo "Expected -s <sampler>:<n_steps> with sampler euler, sde, pndm or dpm; got ${OPTARG}" >&2
                exit 1
            fi
            SOLVERS+=("${OPTARG}")
            ;;
        d)
            CKPT_DIR=${OPTARG}
            ;;
        e)
            # 3,51,90; brackets and spaces are dropped, so "[3, 51, 90]" also works.
            IFS=', ' read -ra epoch_list <<<"${OPTARG//[][]/}"
            for epoch in "${epoch_list[@]}"; do
                if [[ ! ${epoch} =~ ^[0-9]+$ ]]; then
                    echo "Expected -e <epoch>[,<epoch>...] with integer epochs; got ${OPTARG}" >&2
                    exit 1
                fi
                EPOCHS+=("$((10#${epoch}))")
            done
            ;;
        w)
            IFS=', ' read -ra weights_list <<<"${OPTARG}"
            for weights in "${weights_list[@]}"; do
                case "${weights,,}" in
                    none) WEIGHTS+=(none) ;;
                    ema) WEIGHTS+=(EMA) ;;
                    swa) WEIGHTS+=(SWA) ;;
                    *)
                        echo "Expected -w none, EMA or SWA (comma-separated for several); got ${OPTARG}" >&2
                        exit 1
                        ;;
                esac
            done
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))
if ((${#SOLVERS[@]} == 0)); then
    SOLVERS=("${DEFAULT_SOLVER}")
fi

# Prepends the -d/-e/-w selection to the listed checkpoints, so both go through
# parse_checkpoint below. Every (epoch, weights) pair without exactly one match is reported
# before exiting.
if [[ -n ${CKPT_DIR} ]]; then
    if ((${#EPOCHS[@]} == 0)); then
        echo "-d needs -e <epoch>[,<epoch>...] to pick checkpoints" >&2
        exit 1
    fi
    if ((${#WEIGHTS[@]} == 0)); then
        WEIGHTS=(none)
    fi
    if [[ -d ${CKPT_DIR}/ckpts ]]; then
        CKPT_DIR=${CKPT_DIR}/ckpts
    fi
    if [[ ! -d ${CKPT_DIR} ]]; then
        echo "No such checkpoint directory: ${CKPT_DIR}" >&2
        exit 1
    fi
    selected=()
    n_unmatched=0
    for epoch in "${EPOCHS[@]}"; do
        for weights in "${WEIGHTS[@]}"; do
            matches=()
            for ckpt in "${CKPT_DIR}"/epoch="$(printf '%03d' "${epoch}")"-*.ckpt; do
                # The weights suffix must match exactly, so EMA skips -EMA_decay<d>_epochs<N>.
                if [[ $(basename "${ckpt}" .ckpt) =~ ^epoch=[0-9]+-step=[0-9]+-val_loss=-?[^-]+(-(.+))?$ &&
                    ${BASH_REMATCH[2]:-none} == "${weights}" ]]; then
                    matches+=("${ckpt}")
                fi
            done
            if ((${#matches[@]} == 1)); then
                selected+=("${matches[0]}")
            else
                echo "Found ${#matches[@]} checkpoints for epoch ${epoch} with ${weights} weights" \
                    "in ${CKPT_DIR}${matches[*]:+: ${matches[*]}}" >&2
                n_unmatched=$((n_unmatched + 1))
            fi
        done
    done
    if ((n_unmatched > 0)); then
        exit 1
    fi
    set -- "${selected[@]}" "$@"
elif ((${#EPOCHS[@]} > 0 || ${#WEIGHTS[@]} > 0)); then
    echo "-e and -w pick checkpoints from the run dir given with -d" >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_NTASKS:-}" ]]; then
    echo "Run this inside an interactive allocation (salloc); see the header for an example." >&2
    exit 1
fi

# Sets CKPT, RUN_DIR and OUTPUT_STEM (the output name without its solver tag) from
# <run_dir>/ckpts/<checkpoint>.ckpt; exits on malformed paths before anything is launched.
parse_checkpoint() {
    local run_name ckpt_stem ckpt_suffix run_tag ckpt_tag weights_tag=""
    CKPT=$(realpath -e "$1")
    RUN_DIR=$(dirname "$(dirname "${CKPT}")")
    run_name=$(basename "${RUN_DIR}")
    if [[ $(basename "$(dirname "${CKPT}")") != "ckpts" || ! -f "${RUN_DIR}/config.yaml" ]]; then
        echo "Expected <run_dir>/ckpts/<checkpoint>.ckpt with <run_dir>/config.yaml; got ${CKPT}" >&2
        exit 1
    fi
    # Run tag is the run dir name minus its trailing _<YYYYMMDD>-T<HHMMSS> timestamp:
    # v6_6_clas12_RGA_particle_20261012-T150456 -> v6_6_clas12_RGA_particle.
    if [[ ${run_name} =~ ^(.+)_[0-9]{8}-T[0-9]{6}$ ]]; then
        run_tag=${BASH_REMATCH[1]}
    else
        echo "Cannot find a _<YYYYMMDD>-T<HHMMSS> timestamp at the end of run dir name: ${run_name}" >&2
        exit 1
    fi
    # Checkpoints are epoch=<NNN>-step=<N>-val_loss=<x>[-<suffix>], last[-<suffix>] or
    # sigterm; the latter two carry no epoch, so they are tagged by name.
    ckpt_stem=$(basename "${CKPT}" .ckpt)
    if [[ ${ckpt_stem} =~ ^epoch=([0-9]+)- ]]; then
        ckpt_tag="epoch${BASH_REMATCH[1]}"
    else
        ckpt_tag=${ckpt_stem%%-*}
    fi
    # Averaged-weight checkpoints end in -EMA, -SWA or -EMA_decay<d>_epochs<N>.
    ckpt_suffix=${ckpt_stem##*-}
    case "${ckpt_suffix}" in
        EMA* | SWA*) weights_tag="_${ckpt_suffix}" ;;
    esac
    OUTPUT_STEM="${run_tag}_particleNetTest_${ckpt_tag}${weights_tag}"
}

# One entry per (checkpoint, solver) test.
CKPTS=()
RUN_DIRS=()
OUTPUT_NAMES=()
SAMPLERS=()
STEPS=()
declare -A OUTPUT_OWNER
for arg in "$@"; do
    parse_checkpoint "${arg}"
    for solver in "${SOLVERS[@]}"; do
        sampler=${solver%%:*}
        n_steps=${solver#*:}
        if [[ ${sampler} == dpm ]]; then
            output_name="${OUTPUT_STEM}_s${n_steps}"
        else
            output_name="${OUTPUT_STEM}_${sampler}_s${n_steps}"
        fi
        output_key="${RUN_DIR}/${output_name}"
        if [[ -n "${OUTPUT_OWNER[${output_key}]:-}" ]]; then
            echo "Both ${OUTPUT_OWNER[${output_key}]} and ${CKPT} would write ${output_name} in ${RUN_DIR}" >&2
            exit 1
        fi
        OUTPUT_OWNER[${output_key}]=${CKPT}
        CKPTS+=("${CKPT}")
        RUN_DIRS+=("${RUN_DIR}")
        OUTPUT_NAMES+=("${output_name}")
        SAMPLERS+=("${sampler}")
        STEPS+=("${n_steps}")
    done
done

MAX_PARALLEL=${SLURM_NTASKS}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}
# Per-CPU memory for each step. A step without a memory request takes all of the job's
# memory on its node, which would serialize the steps. salloc exports SLURM_MEM_PER_CPU /
# SLURM_MEM_PER_NODE only when --mem-per-cpu / --mem is given; otherwise use the GPU-node
# share of 229802 MB / 128 CPUs, which is also the shared-QOS cap (MaxMemPerCPU=1796).
if [[ -n "${SLURM_MEM_PER_CPU:-}" ]]; then
    MEM_PER_CPU_MB=${SLURM_MEM_PER_CPU}
elif [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
    MEM_PER_CPU_MB=$((SLURM_MEM_PER_NODE / SLURM_CPUS_ON_NODE))
else
    MEM_PER_CPU_MB=$((229802 / 128))
fi
# srun adds salloc's exported --gpus / --gpus-per-node to the step request, which would
# give every step all of the job's GPUs; drop them so each step gets one. The postprocess
# step does not use its GPU, but requesting the same resources keeps one slot per GPU.
SRUN_STEP=(
    env -u SLURM_GPUS -u SLURM_GPUS_PER_NODE
    srun --nodes=1 --ntasks=1 --exact --gpus-per-task=1
    --cpus-per-task="${CPUS_PER_TASK}" --mem-per-cpu="${MEM_PER_CPU_MB}M"
)

# Run from the repository root with the parnassus-core environment active. DATA is
# the directory holding v8/v8_test_filtered_truth.root.
: "${DATA:?export DATA=<directory holding v8/v8_test_filtered_truth.root>}"

export OMP_NUM_THREADS=8

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

# Input to the particle test, and the reference for postprocessing its predictions.
TEST_DATA="${DATA}/v8/v8_test_filtered_truth.root"

STATUS_DIR=$(mktemp -d)
trap 'rm -rf "${STATUS_DIR}"' EXIT

run_eval() {
    local i=$1 rc=0 stage=test
    local log="${LOG_DIR}/${OUTPUT_NAMES[i]}_${SLURM_JOB_ID}.log"
    # PredictionWriter names its file <eval_output_name>_<n_events / 1000>k.root.
    local test_file="${RUN_DIRS[i]}/${OUTPUT_NAMES[i]}_$((N_TEST / 1000))k.root"
    echo "[start $(date +%T)] ${OUTPUT_NAMES[i]} <- ${CKPTS[i]}"
    "${SRUN_STEP[@]}" \
        python -m parnassus_core.experiments.clas12_RGA.main test \
        -c "${RUN_DIRS[i]}/config.yaml" \
        --ckpt_path "${CKPTS[i]}" \
        --data.test_path "${TEST_DATA}" \
        --data.test_batch_size 1000 \
        --data.num_test "${N_TEST}" \
        --data.recalculate_ip true \
        --data.eval_output_name "${OUTPUT_NAMES[i]}" \
        --model.sampler_type "${SAMPLERS[i]}" \
        --model.n_steps "${STEPS[i]}" \
        --matmul_precision high \
        --trainer.num_nodes 1 \
        --trainer.devices 1 \
        --trainer.strategy auto \
        --trainer.callbacks+=parnassus_core.flow.callbacks.PredictionWriter \
        >"${log}" 2>&1 || rc=$?
    if ((rc == 0)); then
        stage=postprocess
        echo "[post  $(date +%T)] ${OUTPUT_NAMES[i]} <- ${test_file}"
        "${SRUN_STEP[@]}" \
            python -m parnassus_core.evaluation.postprocess_eval \
            -i "${test_file}" \
            -d "${TEST_DATA}" \
            -o "${RUN_DIRS[i]}/${OUTPUT_NAMES[i]}_postprocess_dr_0_3.root" \
            -n "${N_TEST}" \
            -c "${POSTPROCESS_CONFIG}" \
            >>"${log}" 2>&1 || rc=$?
    fi
    if ((rc == 0)); then
        echo 0 >"${STATUS_DIR}/${i}"
    else
        echo "${stage} exit ${rc}" >"${STATUS_DIR}/${i}"
    fi
    echo "[done  $(date +%T)] ${OUTPUT_NAMES[i]} ${stage} exit=${rc} (log: ${log})"
}

echo "Running ${#CKPTS[@]} test(s) (${#SOLVERS[@]} solver(s) per checkpoint), at most ${MAX_PARALLEL} at a time" \
    "(${CPUS_PER_TASK} CPUs, ${MEM_PER_CPU_MB} MB/CPU, 1 GPU each)"

for i in "${!CKPTS[@]}"; do
    while (($(jobs -rp | wc -l) >= MAX_PARALLEL)); do
        wait -n || true
    done
    run_eval "${i}" &
done
wait || true

n_failed=0
echo "Summary:"
for i in "${!CKPTS[@]}"; do
    rc=$(cat "${STATUS_DIR}/${i}" 2>/dev/null || echo "missing")
    if [[ ${rc} == 0 ]]; then
        status="ok"
    else
        status="FAILED (${rc})"
        n_failed=$((n_failed + 1))
    fi
    printf '  %-50s %s\n' "${OUTPUT_NAMES[i]}" "${status}"
done
if ((n_failed > 0)); then
    echo "${n_failed} of ${#CKPTS[@]} evaluation(s) failed; logs: ${LOG_DIR}/*_${SLURM_JOB_ID}.log" >&2
    exit 1
fi
echo "Finished: $(date)"
