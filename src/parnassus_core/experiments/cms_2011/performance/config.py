import yaml
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Self


def init_latex():
    """Initialize the LaTeX environment for the plotting."""
    import os

    if Path("/storage/agrp/dmitrykl/latex/texlive/2024/bin/x86_64-linux").exists():
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/bin/x86_64-linux"
        )

        # setup libgd
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/libgd-2.3.3/bin"
        )
        # setup dvipng
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/bin"
        )
        os.environ["LD_LIBRARY_PATH"] = (
            os.environ["LD_LIBRARY_PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/libgd-2.3.3/lib:/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/kpathsea_aux/lib"
        )
    else:
        pass


class Source(Enum):
    """Enumeration for the source type of the data.
    This is used to distinguish between different data sources
    such as Flow Matching, Delphes, Diffusion, Truth, and CMS Pflow.

    Attributes:
        FM: Flow Matching data source.
        DF: Delphes data source.
        DL: Diffusion data source.
        TR: Truth data source.
        PF: CMS Pflow data source.
    """

    FM = "fm"
    DF = "df"
    DL = "dl"
    TR = "tr"
    PF = "pf"


class Sample(Enum):
    """Enumeration for the different samples used in the performance evaluation.
    Attributes:
        H4LEP: Sample for H4Lep events.
        TTBAR: Sample for TTBar events.
        QCD_470: Sample for QCD events with 470 GeV.
        QCD_600: Sample for QCD events with 600 GeV.
    """

    H4LEP = "h4lep"
    TTBAR = "ttbar"
    QCD_470 = "qcd_470"
    QCD_600 = "qcd_600"


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
class PerformanceConfig:
    sample_dict: dict[Sample, list[PflowDataConfig]]

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        sample_dict = {}
        for sample_name, pflow_data_list in config["samples"].items():
            sample = Sample[sample_name.upper()]
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
