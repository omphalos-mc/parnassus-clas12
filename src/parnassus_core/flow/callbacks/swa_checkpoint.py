"""Write the Stochastic Weight Averaging (SWA) model to its own checkpoint.

Lightning's ``StochasticWeightAveraging`` keeps the averaged weights only in its
callback state and copies them into the module at the end of training, after the
last regular checkpoint has been written. This callback saves that averaged model
as ``<final checkpoint name>-SWA.ckpt`` next to the regular checkpoints, mirroring
the ``-EMA.ckpt`` naming of ``EMAModelCheckpoint``.
"""

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class SWACheckpoint(Callback):
    """Save the SWA-averaged model when training ends.

    Requires a ``StochasticWeightAveraging`` callback and a checkpoint callback
    (whose directory and filename template are reused). Does nothing if either is
    missing, if SWA never averaged, or under ``fast_dev_run``.
    """

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        swa = next(
            (
                cb
                for cb in trainer.callbacks
                if isinstance(cb, StochasticWeightAveraging)
            ),
            None,
        )
        ckpt_cb = trainer.checkpoint_callback
        if trainer.fast_dev_run or swa is None or ckpt_cb is None:
            return
        swa_state = swa.state_dict()
        if swa_state["n_averaged"] == 0:
            rank_zero_info(
                "[SWACheckpoint] SWA never averaged; no SWA checkpoint written."
            )
            return

        # Same weights SWA itself transfers at train end, so callback order is irrelevant.
        pl_module.load_state_dict(swa_state["average_model_state"])

        # The trainer has already advanced current_epoch past the final epoch.
        metrics = {
            **trainer.callback_metrics,
            "epoch": torch.tensor(trainer.current_epoch - 1),
            "step": torch.tensor(trainer.global_step),
        }
        ext = ckpt_cb.FILE_EXTENSION
        filepath = ckpt_cb.format_checkpoint_name(metrics).replace(ext, f"-SWA{ext}")
        rank_zero_info(
            f"[SWACheckpoint] Saving SWA weights (n_averaged={swa_state['n_averaged']}) "
            f"to {filepath}"
        )
        trainer.save_checkpoint(filepath)
