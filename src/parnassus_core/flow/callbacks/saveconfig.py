import os
import shutil
import socket
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.cli import LightningArgumentParser, Namespace


class SaveConfig(Callback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
    ) -> None:
        super().__init__()
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.already_saved = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved or trainer.fast_dev_run or stage != "fit":
            return
        self.make_path = Path
        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None

        if trainer.is_global_zero:
            print("-" * 80)
            print(f"log dir: {log_dir!r}")
            print("-" * 80)

            if log_dir:
                log_dir = Path(log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                config_path = self.make_path(log_dir / self.config_filename)
                self.save_config(config_path)
                self.save_metadata(trainer, log_dir, pl_module)

            if isinstance(trainer.logger, CometLogger):
                for file in log_dir.glob("*.yaml"):
                    trainer.logger.experiment.log_asset(file)
                base_dir = Path(__file__).parents[3]
                for file in (base_dir / "src").glob("**/*.py"):
                    trainer.logger.experiment.log_code(file)

            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved, src=0)

    def save_config(self, config_path):
        # the `log_dir` needs to be created as we rely on the logger
        # to do it usually but it hasn't logged anything at this point
        config_path.parent.mkdir(parents=True, exist_ok=True)
        print("-" * 100)
        print(f"Created output dir {config_path.parent}")
        print("-" * 100, "\n")

        # copy the norm dict
        sd_path = self.make_path(
            config_path.parent / self.make_path(self.config.data.scale_dict).name
        )
        self.config.data.scale_dict = self.write_file(
            self.config.data.scale_dict, sd_path
        )
        # copy the var dict
        vd_path = self.make_path(
            config_path.parent / self.make_path(self.config.data.var_dict).name
        )
        self.config.data.var_dict = self.write_file(self.config.data.var_dict, vd_path)

        # write config
        self.write_yaml_file(self.config, config_path)

    def write_file(self, file, store_path):
        """Writes the file on S3 at the store_path."""
        shutil.copyfile(file, store_path)
        outpath = str(store_path.resolve())
        return outpath

    def write_yaml_file(self, yaml_file, store_path):
        self.parser.save(
            yaml_file,
            str(store_path),
            skip_none=False,
        )

    def save_metadata(self, trainer, log_dir: Path, pl_module: LightningModule) -> None:
        logger = trainer.logger
        datamodule = trainer.datamodule

        # Metadata that we want to log
        meta = {
            "num_train": len(datamodule.train_dataloader().dataset),
            "num_val": len(datamodule.val_dataloader().dataset),
            "batch_size": datamodule.train_dataloader().batch_size,
            "trainable_params": sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad
            ),
            "num_gpus": trainer.num_devices,
            "gpu_ids": trainer.device_ids,
            "num_workers": datamodule.train_dataloader().num_workers,
            "torch_version": str(torch.__version__),
            "lightning_version": str(L.__version__),
            "cuda_version": torch.version.cuda,
            "hostname": socket.gethostname(),
            "out_dir": logger.save_dir,
            "log_url": getattr(logger.experiment, "url", None),
        }

        if hasattr(trainer, "timestamp"):
            meta["timestamp"] = trainer.timestamp

        # Log the SLURM info if its present
        if "SLURM_JOB_ID" in os.environ:
            meta["slurm_job_id"] = "SLURM_" + str(os.environ["SLURM_JOB_ID"])

        # Log the metadata to the logger if we are using one
        logger = pl_module.logger
        if logger:
            logger.log_hyperparams(meta)

        # Save the metadata locally to a YAML file
        meta_path = log_dir / "metadata.yaml"
        with meta_path.open("w") as file:
            yaml.dump(meta, file, sort_keys=False)
