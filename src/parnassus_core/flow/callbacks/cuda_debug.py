import torch
from lightning.pytorch.callbacks import Callback


class CUDADebugCallback(Callback):
    """Callback that synchronizes CUDA after each step to catch errors at their source."""

    @staticmethod
    def _should_barrier(trainer) -> bool:
        return (
            trainer.world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Check for CUDA errors
            torch.cuda.current_stream().synchronize()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.current_stream().synchronize()

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._should_barrier(trainer):
            torch.distributed.barrier()
        print(
            f"[Rank {trainer.global_rank}] Train epoch {trainer.current_epoch} ended successfully",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._should_barrier(trainer):
            torch.distributed.barrier()
        print(
            f"[Rank {trainer.global_rank}] Validation epoch {trainer.current_epoch} ended successfully",
            flush=True,
        )
