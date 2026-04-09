"""Top level training script, powered by the lightning CLI."""

import pathlib
import comet_ml  # noqa: F401
from lightning.pytorch.cli import ArgsType

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.utils.cli import CLI
from parnassus_core.flow.callbacks.saveconfig import SaveConfig
from parnassus_core.data import ParnassusDataModule

config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=FlowLightning,
        datamodule_class=ParnassusDataModule,
        save_config_callback=SaveConfig,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": [f"{config_dir}/base.yaml"]},
        },
    )


if __name__ == "__main__":
    main()
