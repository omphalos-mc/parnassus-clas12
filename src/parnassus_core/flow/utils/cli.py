import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from jsonargparse.typing import register_type
from lightning.pytorch.cli import LightningCLI


# Add support for converting yaml lists to tensors
def serializer(x: torch.Tensor) -> list:
    return x.tolist()


def deserializer(x: list) -> torch.Tensor:
    return torch.tensor(x)


register_type(torch.Tensor, serializer, deserializer)


def get_best_epoch(config_path: Path) -> Path:
    """Find the best perfoming epoch.

    Parameters
    ----------
    config_path : Path
        Path to saved training config file.

    Returns
    -------
    Path
        Path to best checkpoint for the training run.
    """
    ckpt_dir = Path(config_path.parent / "ckpts")
    print(
        f"No --ckpt_path specified, looking for best checkpoint in {ckpt_dir.resolve()!r}"
    )
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir.resolve()!r}")
    exp = r"(?<=loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, Path(ckpt).name)[0]) for ckpt in ckpts]
    ckpt = ckpts[np.argmin(losses)]
    print(f"Using checkpoint {ckpt.resolve()!r}")
    return ckpt


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument(
            "--name", default="flow", help="Name for this training run."
        )
        parser.add_argument(
            "--move_files_temp",
            default=None,
            help="Temporary directory to rsync data files to before training.",
        )
        parser.link_arguments("name", "trainer.logger.init_args.name")
        parser.link_arguments("name", "trainer.logger.init_args.experiment_name")
        parser.link_arguments("name", "model.name")
        parser.link_arguments(
            "trainer.default_root_dir", "trainer.logger.init_args.save_dir"
        )

    def _rsync_to_temp(self, source_path: str, temp_dir: str) -> str:
        """Rsync a file to the temporary directory and return the new path."""
        # Create temp directory if it doesn't exist
        temp_path_obj = Path(temp_dir)
        temp_path_obj.mkdir(parents=True, exist_ok=True)

        # Get filename and construct temp path
        filename = Path(source_path).name
        temp_path = temp_path_obj / filename

        # Rsync the file if it doesn't already exist in temp
        if not temp_path.exists():
            print(f"\n{'=' * 80}")
            print(f"Starting rsync: {source_path}")
            print(f"         -> to: {temp_path}")
            print(f"{'=' * 80}")
            cmd = ["rsync", "-avh", "--progress", source_path, str(temp_path)]
            subprocess.run(cmd, check=True)
            print(f"{'=' * 80}")
            print(f"Rsync completed: {temp_path}")
            print(f"{'=' * 80}\n")
        else:
            print(f"File already exists in temp: {temp_path}")

        return str(temp_path)

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]

        # Handle rsync of data files to temp directory before instantiation
        move_files_temp = sc.get("move_files_temp")
        if move_files_temp and self.subcommand in ("fit", "test"):
            if not sc.get("trainer.fast_dev_run", False):
                print(
                    f"\nRsyncing data files to temporary directory: {move_files_temp}"
                )
                if self.subcommand == "fit":
                    train_path = sc["data.train_path"]
                    valid_path = sc["data.valid_path"]
                    if train_path == valid_path:
                        print(
                            "Train and validation paths are the same, rsyncing once for both"
                        )
                        temp_path = self._rsync_to_temp(train_path, move_files_temp)
                        sc["data.train_path"] = temp_path
                        sc["data.valid_path"] = temp_path
                    else:
                        sc["data.train_path"] = self._rsync_to_temp(
                            train_path, move_files_temp
                        )
                        sc["data.valid_path"] = self._rsync_to_temp(
                            valid_path, move_files_temp
                        )
                elif self.subcommand == "test":
                    test_path = sc["data.test_path"]
                    sc["data.test_path"] = self._rsync_to_temp(
                        test_path, move_files_temp
                    )

        if self.subcommand == "fit":
            if sc["trainer.fast_dev_run"]:
                n_batches = int(sc["trainer.fast_dev_run"]) + 1
                sc["trainer.logger.init_args.online"] = False
                sc["data.num_train"] = int(n_batches * sc["data.train_batch_size"])
                sc["data.num_valid"] = int(3 * sc["data.valid_batch_size"])

            # Get timestamped output dir for this run
            timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")  # noqa: DTZ005
            log = "trainer.logger"
            name = sc["name"]
            log_dir = Path(sc["trainer.default_root_dir"])

            # Handle case where we re-use an existing config: use parent of timestampped dir
            try:
                datetime.strptime(log_dir.name.split("_")[-1], "%Y%m%d-T%H%M%S")  # noqa: DTZ007
                log_dir = log_dir.parent
            except ValueError:
                pass

            # Set the timestampped dir
            dirname = f"{name}_{timestamp}"
            log_dir_timestamp = str(Path(log_dir / dirname).resolve())
            sc["trainer.default_root_dir"] = log_dir_timestamp
            if sc[log]:
                sc.trainer.logger.dict_kwargs = {
                    "save_dir": log_dir_timestamp,
                    "name": name,
                    "experiment_key": os.environ.get("COMET_EXP_ID", None),
                }

        if self.subcommand == "test":
            # Modify callbacks when testing
            self.save_config_callback = None
            sc["trainer.logger"] = False
            for c in sc["trainer.callbacks"]:
                if hasattr(c, "init_args") and hasattr(c.init_args, "refresh_rate"):
                    c.init_args.refresh_rate = 1
            # Ensure only one device is used for testing
            n_devices = sc["trainer.devices"]
            if (isinstance(n_devices, str | int)) and int(n_devices) > 1:
                print("Setting --trainer.devices=1")
                sc["trainer.devices"] = "1"
            if isinstance(n_devices, list) and len(n_devices) > 1:
                raise ValueError("Testing requires --trainer.devices=1")

    def after_instantiate_classes(self) -> None:
        """After instantiating classes, set the checkpoint path if not provided."""

        sc = self.config[self.subcommand]
        if self.subcommand == "test" and not self.trainer.ckpt_path:
            ckpt_path = sc["ckpt_path"] or get_best_epoch(
                Path(sc["config"][0].rel_path)
            )
            self.trainer.ckpt_path = ckpt_path
            self.trainer.ckpt_dir = Path(ckpt_path).parent
            self.trainer.ckpt_name = str(Path(ckpt_path).stem)
