from typing import Self
import yaml
from pathlib import Path
from dataclasses import dataclass, field

from rich import print as rprint
from parnassus_core.flow.sampling import SamplerType

DEFAULT_MAX_PARTICLES = 400


@dataclass
class ModelConfig:
    ckpt_path: str | Path
    sampler_type: SamplerType = "euler"
    n_steps: int = 25
    sampler_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.sampler_type not in SamplerType.__args__:
            raise ValueError(
                f"Invalid sampler type: {self.sampler_type}. "
                f"Expected one of {SamplerType.__args__}."
            )
        self.ckpt_path = Path(self.ckpt_path)
        assert self.ckpt_path.exists(), (
            f"Checkpoint path {self.ckpt_path} does not exist."
        )
        assert self.n_steps > 0, "Number of steps must be a positive integer."


@dataclass
class DatasetConfig:
    name: str
    path: str
    n_events: int

    def __post_init__(self):
        assert Path(self.path).exists(), f"Dataset path {self.path} does not exist."
        assert self.n_events > 0, "Number of events must be a positive integer."


@dataclass
class DataConfig:
    max_particles: int
    datasets: list[DatasetConfig]
    top_folder: str | Path = ""

    @classmethod
    def from_dict(cls, data_dict: dict) -> Self:
        top_folder = Path(data_dict.get("top_folder", ""))
        max_particles = data_dict.get("max_particles", DEFAULT_MAX_PARTICLES)
        datasets = []
        for ds in data_dict.get("datasets", []):
            datasets.append(
                DatasetConfig(
                    name=ds["name"],
                    path=top_folder / ds["path"],
                    n_events=ds["n_events"],
                )
            )

        return cls(max_particles, datasets=datasets, top_folder=top_folder)


@dataclass(frozen=True)
class EvaluationVarConfig:
    """Parsed output of a YAML variable config file."""

    fs_vars: dict[str, list[str]]
    ctxt_vars: list[str]
    ctxt_global_vars: list[str]
    event_vars: list[str] = field(default_factory=lambda: ["eventNumber"])

    def __post_init__(self):
        assert "event" in self.fs_vars, "fs_vars must contain an 'event' key."
        assert "particle" in self.fs_vars, "fs_vars must contain a 'particle' key."

    @staticmethod
    def from_yaml(var_dict: str) -> "EvaluationVarConfig":
        """Parse YAML variable config into a :class:`VarConfig`.

        Handles both new ``vars_to_load`` dict and legacy
        ``{coll}_vars_to_load`` format.

        Example YAML::

            event_vars: [eventNumber, Sim_PV_x, Sim_PV_y, Sim_PV_z]
            variable_remapping:
              Sim_PV_x: pv_x
              Sim_PV_y: pv_y
              Sim_PV_z: pv_z
        """
        with open(var_dict) as f:
            cfg = yaml.safe_load(f)

        return EvaluationVarConfig(
            ctxt_vars=cfg["ctxt_vars"],
            fs_vars=cfg["fs_vars"],
            ctxt_global_vars=cfg["ctxt_global_vars"],
            event_vars=cfg.get("event_vars", ["eventNumber"]),
        )


@dataclass
class EvaluationConfig:
    scale_dict: str
    var_dict: str
    eval_path: Path
    var_config: EvaluationVarConfig
    data_config: DataConfig
    event_model: ModelConfig
    particle_model: ModelConfig
    impact_model: ModelConfig | None = None
    recalculate_ip: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Self:
        yaml_path = Path(yaml_path)
        assert yaml_path.exists(), f"YAML config file {yaml_path} does not exist."

        with open(yaml_path, "r") as file:
            config_data = yaml.safe_load(file)

        eval_path = config_data.get("eval_path", "")
        if isinstance(eval_path, str):
            eval_path = Path(eval_path)
        if eval_path.exists():
            assert eval_path.is_dir(), f"Eval path {eval_path} must be a directory."
            rprint(
                f"[bold yellow]WARNING:[/bold yellow] Using existing eval path: {eval_path.absolute()}"
            )
        else:
            rprint(
                f"[bold green]Creating eval path:[/bold green] {eval_path.absolute()}"
            )
            eval_path.mkdir(parents=True, exist_ok=True)

        scale_dict = config_data["scale_dict"]
        var_dict = config_data["var_dict"]

        data_config = DataConfig.from_dict(config_data.get("data", {}))
        event_model = ModelConfig(**config_data["event_model"])
        particle_model = ModelConfig(**config_data["particle_model"])
        impact_model = (
            ModelConfig(**config_data["impact_model"])
            if "impact_model" in config_data
            else None
        )

        var_config = EvaluationVarConfig.from_yaml(var_dict)

        return cls(
            eval_path=eval_path,
            scale_dict=scale_dict,
            var_dict=var_dict,
            var_config=var_config,
            data_config=data_config,
            event_model=event_model,
            particle_model=particle_model,
            impact_model=impact_model,
            recalculate_ip=config_data.get("recalculate_ip", False),
        )
