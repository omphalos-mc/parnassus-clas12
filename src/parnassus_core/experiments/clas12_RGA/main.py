"""Top level training script, powered by the lightning CLI."""

import pathlib
import comet_ml  # noqa: F401
from lightning.pytorch.cli import ArgsType

from parnassus_core.flow.models.wrapper import FlowLightning
from parnassus_core.flow.utils.cli import CLI
from parnassus_core.flow.callbacks.saveconfig import SaveConfig
from parnassus_core.data import ParnassusDataModule

config_dir = pathlib.Path(__file__).parent / "configs"


def _skip_parse_ckpt_path(self) -> None:
    """No-op override of ``LightningCLI._parse_ckpt_path`` (see note below)."""
    return None


# PyTorch >=2.6 defaults ``torch.load(weights_only=True)``, which refuses the custom
# classes pickled into our (trusted, self-produced) checkpoints (e.g. Lion, FlowMatcher).
# ``LightningCLI._parse_ckpt_path`` loads the ckpt with an *explicit* ``weights_only=True``
# solely to merge its saved ``hyper_parameters`` into the config on resume -- redundant
# here since the submit scripts always pass the full config via ``-c``. That explicit True
# cannot be overridden by ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`` (that env var only affects
# callers that leave ``weights_only`` unset), so we skip the pre-parse entirely. The real
# resume load (trainer checkpoint connector, ``weights_only`` unset) is handled by
# ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` exported in the submit scripts.
CLI._parse_ckpt_path = _skip_parse_ckpt_path


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
