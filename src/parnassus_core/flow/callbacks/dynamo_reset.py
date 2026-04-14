"""Callback to periodically reset torch dynamo cache to prevent memory issues."""

import torch
from lightning.pytorch.callbacks import Callback


class DynamoResetCallback(Callback):
    """Reset torch.compile cache periodically to prevent memory corruption.

    FlexAttention uses torch.compile internally, and the dynamo cache can grow
    over many epochs, eventually causing issues with DDP gradient synchronization.

    Parameters
    ----------
    reset_every_n_epochs : int
        Reset the dynamo cache every N epochs. Default is 50.
    """

    def __init__(self, reset_every_n_epochs: int = 50):
        super().__init__()
        self.reset_every_n_epochs = reset_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        """Reset dynamo cache periodically."""
        if (trainer.current_epoch + 1) % self.reset_every_n_epochs == 0:
            # Synchronize CUDA before reset
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Reset dynamo cache
            torch._dynamo.reset()

            if trainer.is_global_zero:
                print(
                    f"[DynamoResetCallback] Reset dynamo cache at epoch {trainer.current_epoch + 1}"
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Ensure CUDA operations complete before worker cleanup."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Add a barrier to ensure all ranks are synchronized
        if trainer.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
