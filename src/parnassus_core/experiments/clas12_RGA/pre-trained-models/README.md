# Pre-trained models

The checkpoints behind the paper results:

| File | Model |
|---|---|
| `event_model.ckpt` | Event model, SWA weights at epoch 109 (9 MB) |
| `particle_model.ckpt` | Particle model, SWA weights at epoch 109 (35 MB) |
| `pre-filter_model.ckpt` | scatE pre-filter classifier, trained on 10M events of `v8/pre-filter_train_data_v8.root` (0.2 MB); see [`../scatE_classifier/README.md`](../scatE_classifier/README.md) |

`evaluation/configs/config.yaml` (`ckpt_path`) and `evaluation/scatE_filter.py` (`--ckpt`)
read them from here, with paths relative to the repository root.

The event and particle checkpoints are weights only, in the layout Lightning writes with
`save_weights_only=True`: the weights, the hyperparameters and the epoch and step counters,
without the optimizer, SWA and EMA state that the training checkpoints also carry. They give
the same generated events as the full training checkpoints, bit for bit, and work with
`eval.py`, the speed test, the ONNX export and the Lightning `test` subcommand. Training
cannot be resumed from them.
