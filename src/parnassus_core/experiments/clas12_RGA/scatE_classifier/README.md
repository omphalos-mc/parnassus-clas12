# scatE classifier

The CLAS12 event and particle models are trained only on events in which the scattered
electron was reconstructed (`pflow_scatE == 1`). To run the full chain on generated
events, that selection has to be simulated too. This classifier learns it: a small MLP
maps the truth `(p, theta, phi)` of the scattered electron to `P(pflow_scatE == 1)`,
a reconstruction efficiency in truth kinematics.

`eval.py` applies the trained model while it reads the input, when the eval config has a
`scatE_filter` block ([`parnassus_core/evaluation/event_filter.py`](../../../evaluation/event_filter.py)).
Its default `sample` mode keeps each event with probability `P`, which reproduces the true
efficiency because the classifier is well calibrated. See that module's docstring for the
modes; the full-sim reference is written by
[`evaluation/scatE_filter.py`](../evaluation/scatE_filter.py).

## Training

From the repository root:

```bash
python -m parnassus_core.experiments.clas12_RGA.scatE_classifier.train_scatE \
    --root-file <AllEvents>.root [--n-events N] [--patience P] [--outdir DIR]
```

The input is any `evt_tree` file that keeps the events without a reconstructed electron
(`pflow_scatE` is both 0 and 1). The run writes `best_model.ckpt` (with the feature
standardization stored inside it), `config.json`, `metrics.json`, `history.csv` and the
loss, ROC and confusion-matrix plots into `--outdir` (default `trained_models/classifier`).
The eval config reads the paper checkpoint; set `scatE_filter.ckpt_path` to
`trained_models/classifier/best_model.ckpt` to use a new run instead.

## The checkpoint used for the paper results

[`pre-trained-models/pre-filter_model.ckpt`](../pre-trained-models/README.md) is trained on
`v8/pre-filter_train_data_v8.root`.
That sample keeps every event, with and without a reconstructed scattered electron
(19.7% have `pflow_scatE == 1`). It overlaps `v8_train.root`, the event and particle models'
training sample, and shares no events with `v8_test.root`, so the evaluation chain filters
events the classifier has not seen.

| Setting | Value |
|---|---|
| Input | First 10M of the file's 19.91M events, split 7M / 1.5M / 1.5M (seed 0) |
| Model | MLP, 4 × 128 hidden units, ReLU, dropout 0.02 (the best trial of the Optuna search in `MLP.ipynb`, run on an earlier CLAS12 sample) |
| Optimizer | Adam, lr 2.929e-3, batch 8192, 50 epochs (early-stopping patience 10, never reached), best epoch 45 |
| Test split | Accuracy 0.957, F1 0.896, AUC 0.977 (majority-class baseline 0.803) |

To retrain it with the same settings (everything else is the default):

```bash
python -m parnassus_core.experiments.clas12_RGA.scatE_classifier.train_scatE \
    --root-file <parnassus_data>/v8/pre-filter_train_data_v8.root --n-events 10000000
```

## Diagnostics

[`plotter.ipynb`](plotter.ipynb) runs the checkpoint over the held-out `v8/v8_test.root` and draws
the truth-kinematics pair plot, the confusion matrix (with Bernoulli sampling, as the fast
simulation uses the model), the ROC curve, and the efficiency profiles and 2-D closure maps in
`p`, `theta` and `phi`. The efficiency profiles are saved as `Filter_Efficiency.pdf`. Set
`PARNASSUS_DATA` to the directory that holds `v8/`, and `PARNASSUS_FIG_DIR` to change where the
figure is written (default `plots/paper_figs/`).
