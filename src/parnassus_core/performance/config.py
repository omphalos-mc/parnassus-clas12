import yaml
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Self, Generic, Type, TypeVar, ClassVar


class Source(Enum):
    """Enumeration for the source type of the data.
    This is used to distinguish between different data sources
    such as Flow Matching, Delphes, Diffusion, Truth, and Pflow.

    Attributes:
        FM: Flow Matching data source.
        DF: Delphes data source.
        DL: Diffusion data source.
        TR: Truth data source.
        PF: Pflow data source.
    """

    FM = "fm"
    DF = "df"
    DL = "dl"
    TR = "tr"
    PF = "pf"


class BaseSample(Enum):
    """Base enumeration class for the different samples used in the performance evaluation."""

    pass


SampleT = TypeVar("SampleT", bound=BaseSample)


@dataclass
class PflowDataConfig:
    name: str
    path: str | Path
    pflow_type: Source
    label: str | None = None
    num_events: int | None = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Network path {self.path} does not exist.")
        if self.num_events is not None and self.num_events < 0:
            raise ValueError("num_events must be a non-negative integer or None.")
        if self.label is None:
            self.label = self.name

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        return cls(
            name=config["name"],
            label=config.get("label"),
            path=config["path"],
            pflow_type=Source(config["pflow_type"]),
            num_events=config.get("num_events"),
        )


@dataclass
class PerformanceConfig(Generic[SampleT]):
    sample_dict: dict[SampleT, list[PflowDataConfig]]

    sample_enum_cls: ClassVar[Type[SampleT]] = BaseSample

    @classmethod
    def from_dict(cls, config: dict) -> "PerformanceConfig[SampleT]":
        sample_dict = {}
        for sample_name, pflow_data_list in config["samples"].items():
            sample = cls.sample_enum_cls[sample_name.upper()]
            pflow_data_configs = [
                PflowDataConfig.from_dict(pflow_data) for pflow_data in pflow_data_list
            ]
            sample_dict[sample] = pflow_data_configs
        return cls(sample_dict=sample_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Self:
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"YAML configuration file {yaml_path} does not exist."
            )
        with yaml_path.open() as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)
