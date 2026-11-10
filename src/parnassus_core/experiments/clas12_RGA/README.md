# CLAS12 RGA Fall 2018 inbending

Parnassus fast simulation of the CLAS12 detector at Jefferson Lab, Hall B: the RG-A Fall 2018
inbending configuration, with a 10.6041 GeV electron beam on a proton target. The models are
trained on `clasdis` deep-inelastic-scattering Monte Carlo, comparing the generator-level
particles (`truth`) with the full GEANT4 simulation and CLAS12 reconstruction (`pflow`). The
paper results use the **v8.1** models described here.

## Generation chain

1. **scatE pre-filter** ([`scatE_classifier/`](scatE_classifier/README.md),
   [`evaluation/scatE_filter.py`](evaluation/scatE_filter.py)). A small MLP maps the truth
   `(p, theta, phi)` of the scattered electron to the probability that reconstruction finds it,
   and each event is kept by a Bernoulli draw on that probability. The next two stages are
   trained only on events with a reconstructed scattered electron.
2. **Event model** (`configs/v8/event_v8_1.yaml`). Generates the reconstructed multiplicity
   `npflow`, the scalar momentum sum `ht_p`, the hadronic momentum sums `Pxh` and `Pyh`, and the
   photon and neutron counts `Ng` and `Nn`, conditioned on the truth particles.
3. **Particle model** (`configs/v8/particle_v8_1.yaml`). Generates every reconstructed particle's
   `p`, `theta`, `phi`, `beta`, `vx`, `vy`, `vz` and `pdgId`, conditioned on the truth particles
   and the event-model output.

Particle variables are scaled per `pdgId` (`class_based_scaling`) with
`configs/v8/var_transform_v8.yaml`; the file header records its provenance.

## Layout

```text
clas12_RGA/
├── main.py                 # Lightning CLI entry point (configs/base.yaml is the default config)
├── modules.py              # SANamedNetwork (self-attention block used by the event model)
├── configs/                # base.yaml + v8/ (v8 and v8.1 training configs, var_transform_v8.yaml)
├── scatE_classifier/       # scatE classifier training (train_scatE.py) and notes
├── evaluation/             # scatE_filter.py, speed_test.py, configs/v8/ (eval + postprocess)
├── performance/            # CLAS12 plotting on top of parnassus_core.performance
├── scripts/                # SLURM jobs behind the v8.1 results (see scripts/README.md)
└── notebooks/              # performance_test.ipynb and dihadron_plots.ipynb (paper figures)
```

## Data

The `evt_tree` ROOT files are produced from CLAS12 HIPO files by the separate
[`parnassus_data_prep`](https://github.com/tjayhllstrn/parnassus_data_prep) pipeline. The v8
samples are:

| File | Content |
|---|---|
| `v8/v8_train.root` | Training sample, 59.0M events, all with a reconstructed scattered electron |
| `v8/v8_test.root` | Test sample, 29.91M events, including those without (about 20% have one) |
| `v8/pre-filter_train_data_v8.root` | scatE classifier training sample, 19.91M events with and without a reconstructed scattered electron; overlaps `v8_train.root`, no events in common with `v8_test.root` |
| `v8/v8_test_filtered.root` | `scatE_filter --mode sample` output: the fast-sim input |
| `v8/v8_test_filtered_truth.root` | `scatE_filter --mode truth` output: the full-sim reference |

The trained checkpoints and the postprocessed evaluation output used for the figures are in the
data release (DOI: TODO). Real CLAS12 RGA data are not distributed.

## Training

From the repository root (data paths in `configs/base.yaml` and the stage configs are
site-specific):

```bash
python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c src/parnassus_core/experiments/clas12_RGA/configs/v8/event_v8_1.yaml
python -m parnassus_core.experiments.clas12_RGA.main fit \
    -c src/parnassus_core/experiments/clas12_RGA/configs/v8/particle_v8_1.yaml
```

Add `--trainer.fast_dev_run 1 --trainer.devices 1 --trainer.num_nodes 1 --trainer.strategy auto`
for a one-step check. The v8.1 models were trained with the multi-node jobs in
[`scripts/`](scripts/README.md). The scatE classifier is trained separately; see
[`scatE_classifier/README.md`](scatE_classifier/README.md).

## Evaluation

`scripts/run_evaluation_chain_v8_1.sh` runs the four steps below. The paper run filtered all
29.91M events of `v8_test.root` and evaluated every kept event (`--n-keep` is only a stopping
target, so the file size keeps everything), then ran step 4 by hand on the first 6M joined events:

```bash
sbatch -A <account> src/parnassus_core/experiments/clas12_RGA/scripts/run_evaluation_chain_v8_1.sh \
    --refilter --n-read 29910000 --n-keep 29910000 --suffix v8_filter_29M_test --skip-postprocess
```

By hand, with `DATA` pointing at the directory that holds `v8/`:

```bash
# 1. fast-sim input: keep each event ~ Bernoulli(classifier probability)
python -m parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter \
    -i $DATA/v8/v8_test.root -o $DATA/v8/v8_test_filtered.root --mode sample --seed 0

# 2. full-sim reference: keep the events whose scattered electron was reconstructed
python -m parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter \
    -i $DATA/v8/v8_test.root -o $DATA/v8/v8_test_filtered_truth.root --mode truth

# 3. full-chain generation (event model -> particle model)
python -m parnassus_core.evaluation.eval \
    -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/v8/config_v8_1.yaml \
    -bs 4096 -ne -1 --num_workers 8 --suffix v8_filter_29M_test

# 4. cuts, matching, DIS kinematics, four-vectors; union join with the reference.
#    -n 6000000 keeps the first 6M joined events by eventNumber; -n -1 keeps all of them.
python -m parnassus_core.evaluation.postprocess_eval \
    -i clasdis_v8_1_filtered/clasdis_v8_filtered_eval_5862kv8_filter_29M_test.root \
    -d $DATA/v8/v8_test_filtered_truth.root \
    -o clasdis_v8_1_filtered/postprocess_dr_0_3_v8_filter_29M.root -n 6000000 \
    -c src/parnassus_core/experiments/clas12_RGA/evaluation/configs/v8/postprocess_union.yaml
```

`config_v8_1.yaml` holds the checkpoint paths, the samplers (PNDM with 25 steps for the event
model, DPM with 40 for the particle model) and `data.top_folder`; edit those for your copy.

**Why two filter passes.** The classifier's keep decision is stochastic and independent of what
reconstruction did, so some kept events have no reconstructed scattered electron, and their
`pflow` branches are not a valid reference. `postprocess_eval` therefore joins the generated
events (`-i`) to the truth-mode file (`-d`) by `eventNumber`. With the union join
(`postprocess_union.yaml`), an empty `fastsim` or `pflow` collection marks an event that is not
in that sample, so efficiencies and distributions can both be compared. In the paper run, the
5,862,629 generated and 5,900,918 reference events join to 6,909,889 rows. The postprocessed file
keeps the first 6,000,000 of them (`eventNumber` below 25,971,175): 4,214,491 are in both
samples, 875,932 only in the fast sim, 909,576 only in the full sim, and 1 has no particle
passing the cuts.

Use `--mode threshold` only for comparisons: at the 0.5 operating point it over-counts
reconstructed electrons, while Bernoulli sampling reproduces the true rate. Every filter run prints
a closure report (kept fraction, mean probability, true rate, ROC-AUC).

`evaluation/speed_test.py` (`scripts/run_speed_test_v8_1.sh`) times every stage of the chain.

To test single models instead of the full chain, the Lightning `test` subcommand generates from
the event or particle model alone, conditioned on the true quantities.
`scripts/submit_test_event_v8_parallel.sh` and `scripts/submit_test_particle_v8_parallel.sh` run
it for several checkpoints and solvers in parallel inside one allocation; the particle script
then postprocesses each output with `postprocess_df.yaml`.

## Paper figures

[`notebooks/performance_test.ipynb`](notebooks/performance_test.ipynb) (event, DIS and particle
distributions, residuals and PID) and
[`notebooks/dihadron_plots.ipynb`](notebooks/dihadron_plots.ipynb) (dihadron invariant masses) read
`postprocess_dr_0_3_v8_filter_29M_6000k_test_eta49Eval.root` from `EVAL_DIR`. That defaults to
`clasdis_v8_1_filtered/` in the repository root; set `PARNASSUS_EVAL_DIR` to use the data release
instead. Figures are written to `plots/paper_figs/` (`PARNASSUS_FIG_DIR`). The data comparison in
`dihadron_plots` needs processed RGA data, which are not distributed: set `RGA_DATA_FILE` in the
notebook to run those cells.

## Development history

The informal development branch `train_clas12` adds the analysis tools in `tools/` and the
configs of every model version. The `pre-cleanup` tag keeps the code needed to rerun the v0–v7
models, the oracle scans and the ablation studies.
