"""Checkpoint-on-SIGTERM callback for surviving Slurm preemption.

On the Marlowe ``preempt`` QOS (``PreemptMode=requeue``) a preemption delivers
**SIGTERM** to the job step and then waits ``GraceTime`` (~15 min) before
``SIGKILL``. Lightning's auto-requeue handler only reacts to SIGUSR1, so it
ignores SIGTERM (`Bypassing SIGTERM`) and the ~15-min grace window is otherwise
wasted. This callback catches SIGTERM, flags it, and writes **one** atomic
checkpoint at the next safe batch boundary so a preempted run loses ~nothing.

Design notes (see ``context/marlowe_submission.md`` §4-§5):

- SIGTERM is only ever sent when the job is shutting down (preemption, ``scancel``,
  walltime, node drain, ``kill``), never mid-normal-operation, so saving on it does
  not perturb training.
- **Checkpoint-only, never requeue.** Slurm already requeues on preemption and the
  submit scripts resume via ``--ckpt_path``; if this callback also requeued, a
  ``scancel`` could not actually cancel the job.
- The handler stays minimal (just sets a flag) and **chains the previously
  installed handler**, so Lightning's graceful-shutdown path still runs. The real
  save happens at a batch boundary where all DDP ranks are synchronised.
- The write is **atomic** (temp file + ``os.replace``) so an interrupted write
  (e.g. SIGKILL after the 30 s ``KillWait`` on ``scancel``/walltime) cannot corrupt
  the checkpoint.
"""

import os
import signal
from pathlib import Path

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class SigtermCheckpoint(Callback):
    """Save a checkpoint when the process receives SIGTERM.

    Parameters
    ----------
    filename : str
        Basename (without extension) of the checkpoint written on SIGTERM.
        Default ``"sigterm"`` → ``<log_dir>/ckpts/sigterm.ckpt``. The submit
        scripts include this name in their resume glob.
    """

    def __init__(self, filename: str = "sigterm") -> None:
        super().__init__()
        self.filename = filename
        self._received = False
        self._saved = False
        self._prev_handler = None
        self._dirpath: Path | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit" or trainer.fast_dev_run:
            return
        self._dirpath = Path(trainer.log_dir) / "ckpts"
        # Register on every rank so all ranks reach the collective save together.
        # Chain whatever handler is already installed (Lightning's SignalConnector)
        # so its graceful-shutdown logic still runs.
        self._prev_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame) -> None:
        # Keep this minimal: flag it, then chain the previous handler. The actual
        # save runs at the next batch boundary (a safe point for DDP collectives).
        self._received = True
        if callable(self._prev_handler):
            self._prev_handler(signum, frame)

    def _maybe_save(self, trainer: Trainer) -> None:
        if not self._received or self._saved or trainer.fast_dev_run:
            return
        self._saved = True  # save exactly once, even across later batches
        assert self._dirpath is not None
        final = self._dirpath / f"{self.filename}.ckpt"
        tmp = self._dirpath / f".{self.filename}.ckpt.tmp"
        rank_zero_info(f"[SigtermCheckpoint] SIGTERM received; saving {final}")
        try:
            if trainer.is_global_zero:
                self._dirpath.mkdir(parents=True, exist_ok=True)
            # All ranks participate (barrier + rank-0 disk write); write to a temp
            # path first, then atomically rename so a partial write can't clobber
            # a good checkpoint if SIGKILL lands mid-save.
            trainer.save_checkpoint(tmp)
            if trainer.is_global_zero:
                os.replace(tmp, final)
        except Exception as exc:  # never raise during shutdown
            rank_zero_info(f"[SigtermCheckpoint] save failed: {exc!r}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._maybe_save(trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._maybe_save(trainer)
