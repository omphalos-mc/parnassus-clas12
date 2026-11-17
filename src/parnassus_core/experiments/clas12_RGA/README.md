# CLAS12 RGA Fall 2018 inbending

Parnassus fast simulation of the CLAS12 detector at Jefferson Lab, Hall B: the RG-A Fall 2018
inbending configuration, with a 10.6041 GeV electron beam on a proton target. The models are
trained on `clasdis` deep-inelastic-scattering Monte Carlo, comparing the generator-level
particles (`truth`) with the full GEANT4 simulation and CLAS12 reconstruction (`pflow`). The
paper results use the models described here.

## Generation chain

1. **scatE pre-filter** ([`scatE_classifier/`](scatE_classifier/README.md),
   [`parnassus_core/evaluation/event_filter.py`](../../evaluation/event_filter.py)). A small MLP
   maps the truth `(p, theta, phi)` of the scattered electron to the probability that
   reconstruction finds it, and each event is kept by a Bernoulli draw on that probability. The
   `scatE_filter` block of the eval config applies it while the input is read. The next two
   stages are trained only on events with a reconstructed scattered electron.
2. **Event model** (`configs/event.yaml`). Generates the reconstructed multiplicity
   `npflow`, the scalar momentum sum `ht_p`, the hadronic momentum sums `Pxh` and `Pyh`, and the
   photon and neutron counts `Ng` and `Nn`, conditioned on the truth particles.
3. **Particle model** (`configs/particle.yaml`). Generates every reconstructed particle's
   `p`, `theta`, `phi`, `beta`, `vx`, `vy`, `vz` and `pdgId`, conditioned on the truth particles
   and the event-model output.

Particle variables are scaled per `pdgId` (`class_based_scaling`) with
`configs/var_transform.yaml`; the file header records its provenance.

## Layout

```text
clas12_RGA/
├── main.py                 # Lightning CLI entry point (configs/base.yaml is the default config)
├── modules.py              # SANamedNetwork (self-attention block used by the event model)
├── configs/                # Config Files for training
├── scatE_classifier/       # scatE classifier training (train_scatE.py) and plots (plotter.ipynb)
├── pre-trained-models/     # the paper checkpoints: event, particle and scatE models
├── evaluation/             # scatE_filter.py (full-sim reference), speed_test.py, configs/
├── performance/            # CLAS12 plotting on top of parnassus_core.performance
├── scripts/                # SLURM jobs behind the paper results (see scripts/README.md)
└── notebooks/              # performance_test.ipynb and dihadron_plots.ipynb (paper figures)
```

## Data

The `evt_tree` ROOT files are produced from CLAS12 HIPO files by the separate
[`parnassus_data_prep`](https://github.com/tjayhllstrn/parnassus_data_prep) pipeline (tag
`paper-v1`), which also lists the cuts. The samples are:

| File | Content |
|---|---|
| `v8/v8_train.root` | Training sample, 59.0M events, all with a reconstructed scattered electron |
| `v8/v8_test.root` | Test sample, 29.91M events, including those without (about 20% have one) |
| `v8/pre-filter_train_data_v8.root` | scatE classifier training sample, 19.91M events with and without a reconstructed scattered electron; overlaps `v8_train.root`, no events in common with `v8_test.root` |
| `v8/v8_test_filtered.root` | Classifier-sampled `v8_test.root`: the fast-sim input of the paper run, written by the earlier standalone filter (the chain now samples inside `eval.py`) |
| `v8/v8_test_filtered_truth.root` | `scatE_filter.py` output, the `pflow_scatE == 1` events: the full-sim reference |

The trained checkpoints are in [`pre-trained-models/`](pre-trained-models/README.md). The
postprocessed evaluation output used for the figures is in the data release (DOI: TODO).
Real CLAS12 RGA data are not distributed.

## Training

From the repository root (data paths in `configs/base.yaml` and the stage configs are
site-specific):

```bash
python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c src/parnassus_core/experiments/clas12_RGA/configs/event.yaml
python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c src/parnassus_core/experiments/clas12_RGA/configs/particle.yaml
```

Add `--trainer.fast_dev_run 1 --trainer.devices 1 --trainer.num_nodes 1 --trainer.strategy auto`
for a one-step check. The paper models were trained with the multi-node jobs in
[`scripts/`](scripts/README.md). The scatE classifier is trained separately; see
[`scatE_classifier/README.md`](scatE_classifier/README.md).

## Evaluation

`scripts/run_evaluation_chain.sh` runs the three steps below. The paper run evaluated every
classifier-kept event of all 29.91M events of `v8_test.root` (`--n-events` is only a stopping
target, so the file size keeps everything), then ran step 3 by hand on the first 6M joined events:

```bash
sbatch -A <account> src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain.sh \
    --refilter --n-read 29910000 --n-events 29910000 --suffix _paper --skip-postprocess
```

By hand, with `DATA` pointing at the directory that holds `v8/`:

```bash
# 1. full-sim reference: keep the events whose scattered electron was reconstructed
python -m parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter \
    -i $DATA/v8/v8_test.root -o $DATA/v8/v8_test_filtered_truth.root

# 2. full-chain generation (event model -> particle model); the scatE_filter block of the
#    config keeps each event of v8_test.root ~ Bernoulli(classifier probability)
python -m parnassus_core.evaluation.eval \
    -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/config.yaml \
    -bs 4096 -ne 29910000 --num_workers 8 --suffix _paper

# 3. cuts, matching, DIS kinematics, four-vectors; union join with the reference.
#    -n 6000000 keeps the first 6M joined events by eventNumber; -n -1 keeps all of them.
python -m parnassus_core.evaluation.postprocess_eval \
    -i clasdis_filtered/clasdis_filtered_eval_5862k_paper.root \
    -d $DATA/v8/v8_test_filtered_truth.root \
    -o clasdis_filtered/postprocess_paper.root -n 6000000 \
    -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/postprocess_union.yaml
```

`evaluation/configs/config.yaml` holds the checkpoint paths (the files in `pre-trained-models/`),
the samplers (PNDM with 25 steps for the event model, DPM with 40 for the particle model), the
`scatE_filter` block (classifier checkpoint, mode, seed) and `data.top_folder`; edit
`data.top_folder` for your copy of the data. With the filter on, `n_events` (and `-ne`) counts
kept events: the classifier pass stops reading the input once that many are kept. The paper's
fast-sim input, `v8_test_filtered.root`, was sampled by an earlier standalone version of the same
filter; a rerun selects a statistically equivalent, not identical, event set.

**Why a separate reference.** The classifier's keep decision is stochastic and independent of
what reconstruction did, so some kept events have no reconstructed scattered electron, and their
`pflow` branches are not a valid reference. `postprocess_eval` therefore joins the generated
events (`-i`) to the `pflow_scatE == 1` file (`-d`) by `eventNumber`; its read window (`-ne`)
must cover every event the eval run's classifier pass read. With the union join
(`postprocess_union.yaml`), an empty `fastsim` or `pflow` collection marks an event that is not
in that sample, so efficiencies and distributions can both be compared. In the paper run, the
5,862,629 generated and 5,900,918 reference events join to 6,909,889 rows. The postprocessed file
keeps the first 6,000,000 of them (`eventNumber` below 25,971,175): 4,214,491 are in both
samples, 875,932 only in the fast sim, 909,576 only in the full sim, and 1 has no particle
passing the cuts.

Use `mode: threshold` only for comparisons: at the 0.5 operating point it over-counts
reconstructed electrons, while Bernoulli sampling reproduces the true rate. Every filtered eval
run prints a closure report (kept fraction, mean probability, true rate, ROC-AUC) when the input
carries the `pflow_scatE` label.

**Generator (LUND) input.** `eval.py` also reads the LUND text files event generators write
(`.dat`/`.lund`; a file, a directory of them, or a glob), picked by the path in the config.
The truth particles are built as in `parnassus_data_prep`: final-state particles (`type == 1`)
other than electron neutrinos, with theta of at least 1 degree, and `eventNumber` counts the
events of the input from 0. `evaluation/configs/config_lund.yaml` is the template. Generator
output has no full-simulation counterpart, so there is no reference to postprocess against.

`evaluation/speed_test.py` (`scripts/run_speed_test.sh`) times every stage of the chain.

To test single models instead of the full chain, the Lightning `test` subcommand generates from
the event or particle model alone, conditioned on the true quantities.
`scripts/submit_test_event_parallel.sh` and `scripts/submit_test_particle_parallel.sh` run
it for several checkpoints and solvers in parallel inside one allocation; the particle script
then postprocesses each output with `postprocess_df.yaml`.

## Paper figures

[`notebooks/performance_test.ipynb`](notebooks/performance_test.ipynb) (event, DIS and particle
distributions, residuals and PID) and
[`notebooks/dihadron_plots.ipynb`](notebooks/dihadron_plots.ipynb) (dihadron invariant masses) read
the paper run's postprocessed output from `EVAL_DIR`, under the name it has in the data release:
`postprocess_dr_0_3_v8_filter_29M_6000k_test_eta49Eval.root` (the run's eval output is
`clasdis_v8_filtered_eval_5862kv8_filter_29M_test.root`). `EVAL_DIR` defaults to
`clasdis_filtered/` in the repository root, the `eval_path` of `evaluation/configs/config.yaml`;
set `PARNASSUS_EVAL_DIR` to use the data release instead. Figures are written to `plots/paper_figs/` (`PARNASSUS_FIG_DIR`). The data comparison in
`dihadron_plots` needs processed RGA data, which are not distributed: set `RGA_DATA_FILE` in the
notebook to run those cells.
