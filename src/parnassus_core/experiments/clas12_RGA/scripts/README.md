# SLURM scripts for the paper results

These are the job scripts behind the paper results, written for NERSC Perlmutter
(GPU nodes, `-C gpu`). On another cluster, adapt the `#SBATCH` block and the `srun`
geometry.

| Script | What it runs |
|---|---|
| `submit_train_event.sh` | Trains the event model, `configs/event.yaml` (DDP, multi-node) |
| `submit_train_particle.sh` | Trains the particle model, `configs/particle.yaml` |
| `run_evaluation_chain.sh` | `scatE_filter.py` (full-sim reference) → `eval.py` (scatE-filtered) → `postprocess_eval.py` |
| `run_speed_test.sh` | Times every stage of the inference chain (`evaluation/speed_test.py`) |
| `submit_test_event_parallel.sh` | Lightning `test` of several event-model checkpoints and solvers, in parallel |
| `submit_test_particle_parallel.sh` | The same for the particle model, then `postprocess_eval` with `postprocess_df.yaml` |

## Before submitting

Run everything from the repository root, with the `parnassus-core` environment active
(`sbatch` passes the environment on to the job):

```bash
export DATA=<directory holding v8/v8_train.root and v8/v8_test.root>
mkdir -p logs/slurm                   # the #SBATCH --output/--error directory
sbatch -A <account> src/parnassus_core/experiments/clas12_RGA/scripts/<script>.sh [options]
```

- The training scripts log to Comet online and read `COMET_API_KEY` and
  `COMET_WORKSPACE` from the environment. They take `train_path`/`valid_path` from
  `$DATA`. Pass `--dry_run` to run one training step on the current node (no allocation
  needed).
- The evaluation scripts read the three paper checkpoints from
  [`pre-trained-models/`](../pre-trained-models/README.md) through the `ckpt_path` keys in
  `evaluation/configs/config.yaml`. Edit `data.top_folder` in that config for your copy of
  the data.
- `run_evaluation_chain.sh` writes the full-sim reference into `$DATA/v8/` and skips
  that step when it already exists (`--refilter` rebuilds it).
- The two `submit_test_*_parallel.sh` scripts are run with `bash` inside an interactive
  allocation, not with `sbatch`; their headers give the `salloc` line and the ways to pick
  checkpoints (`-d <run_dir> -e <epochs> -w none|EMA|SWA`, or explicit paths). Outputs are
  written next to the checkpoints, in the training run directory.

## The paper run

```bash
sbatch -A <account> src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain.sh \
    --refilter --n-read 29910000 --n-events 29910000 --suffix _paper --skip-postprocess
python -m parnassus_core.evaluation.postprocess_eval \
    -i clasdis_filtered/clasdis_filtered_eval_5862k_paper.root \
    -d $DATA/v8/v8_test_filtered_truth.root \
    -o clasdis_filtered/postprocess_paper.root -n 6000000 \
    -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/postprocess_union.yaml
```

The job filters and evaluates all 29.91M events of `v8_test.root`; postprocessing then keeps the
first 6M of the 6.91M joined events and writes `clasdis_filtered/postprocess_paper_6000k_test_eta49Eval.root`.
In the data release, the outputs of this run keep the names they were made with:
`clasdis_v8_filtered_eval_5862kv8_filter_29M_test.root` and
`postprocess_dr_0_3_v8_filter_29M_6000k_test_eta49Eval.root`, the file both paper notebooks read.
