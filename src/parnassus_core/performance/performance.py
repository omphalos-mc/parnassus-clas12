import functools
import numpy as np
from tqdm import tqdm

from .config import PflowDataConfig, SampleT, PerformanceConfig, Source
from dataclasses import dataclass, field
from typing import Self, Generic

from .utils import (
    calc_jet_residuals,
    calc_met_ht,
    calc_particle_residuals,
    load_data,
    reshape_phi,
    zero_neutral_vtx,
)
from parnassus_core.utils.matching import match_objects

JET_VARS = ["pt", "eta", "phi", "d2", "c2"]

PARTICLE_VARS = [
    "pt",
    "eta",
    "phi",
    "vx",
    "vy",
    "vz",
    "class",
    # impact parameters
    "d0",
    "d0Error",
    "z0",
    "z0Error",
]


def extract_particle_data(data: dict[str, np.ndarray], split="jet"):
    mask_fn = None
    if split == "jet":
        mask_fn = lambda x: x > -1  # noqa: E731
    elif split == "bkg":
        mask_fn = lambda x: x == -1  # noqa: E731
    new_data = {}
    mask = None
    if mask_fn is not None:
        mask = [mask_fn(x) for x in data["jet_idx"]]
    for key, value in data.items():
        if key in {f"tr_idx_{split}", f"pf_idx_{split}"}:
            new_data[key.replace(f"_{split}", "")] = np.array(
                [el[el > -1] for el in value], dtype=object
            )
            continue
        if key not in PARTICLE_VARS:
            continue
        if mask is not None:
            new_data[key] = np.array(
                [np.array(el)[mask[i]] for i, el in enumerate(value)], dtype=object
            )
        else:
            new_data[key] = value
    return new_data


def run_once(flag_name, message, required_flags: list[str] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            check_flags = getattr(self, "check_flags", True)
            if check_flags and required_flags:
                for flag in required_flags:
                    if not getattr(self, flag, False):
                        raise RuntimeError(
                            f"Required flag {flag} is not set. Cannot run {func.__name__}."
                        )
            if check_flags and getattr(self, flag_name, False):
                print(f"Skipping {func.__name__}: {message}")
                return None
            result = func(self, *args, **kwargs)
            setattr(self, flag_name, True)
            return result

        return wrapper

    return decorator


@dataclass
class Data:
    name: str
    source: Source
    label: str | None = None

    event_number: np.ndarray | None = None

    event_data: dict[str, np.ndarray] = field(default_factory=dict)
    jet_data: dict[str, np.ndarray] = field(default_factory=dict)
    particle_data_all: dict[str, np.ndarray] = field(default_factory=dict)
    particle_data_jet: dict[str, np.ndarray] = field(default_factory=dict)
    particle_data_bkg: dict[str, np.ndarray] = field(default_factory=dict)

    # flags
    check_flags: bool = field(
        default=True, init=False
    )  # Check flags before running methods

    _events_filtered: bool = field(default=False, init=False)

    _event_data_calculated: bool = field(default=False, init=False)
    _event_residuals_calculated: bool = field(default=False, init=False)

    _jets_matched: bool = field(default=False, init=False)
    _jets_residuals_calculated: bool = field(default=False, init=False)

    _particles_residuals_calculated: bool = field(default=False, init=False)
    _impact_paramters_calculated: bool = field(default=False, init=False)

    def __getitem__(self, name):
        """Get the data for the given name."""

        args = name.split("/")
        match args:
            case ["event", *rest]:
                return self.event_data.get(rest[0], None)
            case ["jet", *rest]:
                return self.jet_data.get(rest[0], None)
            case ["particle", collection, *rest]:
                if collection == "all":
                    return self.particle_data_all.get(rest[0], None)
                elif collection == "jet":
                    return self.particle_data_jet.get(rest[0], None)
                elif collection == "bkg":
                    return self.particle_data_bkg.get(rest[0], None)
                else:
                    raise ValueError(f"Unknown particle collection: {collection}")

    @classmethod
    def load_from_dict(
        cls,
        name: str,
        source: Source,
        data_dict: dict[str, np.ndarray],
        event_number: np.ndarray | None = None,
        label: str | None = None,
    ) -> Self:
        jet_data = {
            k.replace("jets_jet_", ""): v
            for k, v in data_dict.items()
            if ("jet_" in k and k != "jet_idx")
        }
        jet_data["phi"] = reshape_phi(jet_data["phi"])
        particle_data_all = extract_particle_data(data_dict, split="all")
        particle_data_jet = extract_particle_data(data_dict, split="jet")
        particle_data_bkg = extract_particle_data(data_dict, split="bkg")
        return cls(
            name=name,
            source=source,
            label=label,
            event_number=event_number,
            jet_data=jet_data,
            particle_data_all=particle_data_all,
            particle_data_jet=particle_data_jet,
            particle_data_bkg=particle_data_bkg,
        )

    @run_once("_events_filtered", "events already filtered")
    def filter_events(self, good_events: np.ndarray):
        print(f"Filtering events for {self.name} with source {self.source}")
        idx_sorted = np.argsort(self.event_number)
        idxs = np.argwhere(
            np.isin(self.event_number[idx_sorted], good_events)
        ).flatten()
        for name in [
            "event_data",
            "jet_data",
            "particle_data_all",
            "particle_data_jet",
            "particle_data_bkg",
        ]:
            data = getattr(self, name)
            if not data:
                continue
            for key, value in data.items():
                data[key] = value[idx_sorted][idxs]
            setattr(self, name, data)
        self.event_number = self.event_number[idx_sorted][idxs]
        print(
            f"Data(name={self.name}, source={self.source}), events filtered: {len(good_events)}"
        )

    @run_once(
        "_event_data_calculated",
        "event data already calculated",
        required_flags=["_events_filtered"],
    )
    def calculate_event_data(self):
        print(f"Calculating event data for {self.name} with source {self.source}")
        self.event_data["njet"] = np.array(
            [len(jet) for jet in self.jet_data.get("pt", [])], dtype=int
        )
        self.event_data["npart"] = np.array(
            [len(particle) for particle in self.particle_data_all.get("pt", [])],
            dtype=int,
        )
        met_x, met_y, ht = calc_met_ht(self.particle_data_all)
        self.event_data["met_x"] = met_x
        self.event_data["met_y"] = met_y
        self.event_data["ht"] = ht

    @run_once(
        "_event_residuals_calculated",
        "event residuals already calculated",
        required_flags=["_event_data_calculated"],
    )
    def calculate_event_residuals(self, other: Self):
        print(
            f"Calculating event residuals for {self.name} with source {self.source}"
            f" against {other.name} with source {other.source}"
        )
        assert self.source != other.source, (
            "Sources must be the same for residuals calculation."
        )
        assert (self.source != Source.TR) and (other.source == Source.TR), (
            "Only truth data can be used for residuals calculation."
        )
        keys = [key for key in self.event_data.keys() if "_res" not in key]
        for key in keys:
            assert key in other.event_data, f"Key {key} not found in other data."
            value = self.event_data[key]
            other_value = other.event_data[key]
            if key in {"njet", "npart"}:
                self.event_data[f"{key}_res"] = value - other_value
            else:
                self.event_data[f"{key}_res"] = (value - other_value) / (
                    other_value + 1e-8
                )

    @run_once(
        "_jets_matched",
        "jets already matched",
        required_flags=["_events_filtered"],
    )
    def match_jets(self, other: Self):
        print(
            f"Matching jets for {self.name} with source {self.source}"
            f" against {other.name} with source {other.source}"
        )
        assert self.source != other.source, "Sources must be the same for jet matching."
        assert (self.source != Source.TR) and (other.source == Source.TR), (
            "Only truth data can be used for jet matching."
        )
        pf_idx, tr_idx = [], []
        for i in tqdm(range(len(self.event_number)), desc="Matching jets"):
            input_eta = self.jet_data["eta"][i]
            input_phi = self.jet_data["phi"][i]
            truth_eta = other.jet_data["eta"][i]
            truth_phi = other.jet_data["phi"][i]
            input_idx, truth_idx, _ = match_objects(
                input_eta,
                input_phi,
                truth_eta,
                truth_phi,
                dr_cut=0.2,
            )
            pf_idx.append(input_idx)
            tr_idx.append(truth_idx)
        self.jet_data["pf_idx"] = np.array(pf_idx, dtype=object)
        self.jet_data["tr_idx"] = np.array(tr_idx, dtype=object)

    @run_once(
        "_jets_residuals_calculated",
        "jet residuals already calculated",
        required_flags=["_jets_matched"],
    )
    def calculate_jet_residuals(self, other: Self):
        print(
            f"Calculating jet residuals for {self.name} with source {self.source}"
            f" against {other.name} with source {other.source}"
        )
        assert self.source != other.source, (
            "Sources must be the same for jet residuals calculation."
        )
        assert (self.source != Source.TR) and (other.source == Source.TR), (
            "Only truth data can be used for jet residuals calculation."
        )
        res_dict = calc_jet_residuals(
            self.jet_data["tr_idx"],
            self.jet_data["pf_idx"],
            self.jet_data,
            other.jet_data,
        )
        for key, value in res_dict.items():
            self.jet_data[f"{key}_res"] = value

    @run_once(
        "_particles_residuals_calculated",
        "particle residuals already calculated",
        required_flags=["_events_filtered"],
    )
    def calculate_particle_residuals(self, other: Self):
        print(
            f"Calculating particle residuals for {self.name} with source {self.source}"
            f" against {other.name} with source {other.source}"
        )
        assert self.source != other.source, (
            "Sources must be the same for particle residuals calculation."
        )
        assert (self.source != Source.TR) and (other.source == Source.TR), (
            "Only truth data can be used for particle residuals calculation."
        )
        for split in ["all", "jet", "bkg"]:
            print("Split: ", split)
            pf_data_dict = getattr(self, f"particle_data_{split}")
            tr_data_dict = getattr(other, f"particle_data_{split}")
            res = calc_particle_residuals(
                pf_data_dict["tr_idx"],
                pf_data_dict["pf_idx"],
                pf_data_dict,
                tr_data_dict,
            )
            for key, value in res.items():
                pf_data_dict[f"{key}_res"] = value

    @run_once(
        "_impact_paramters_calculated",
        "impact parameters already calculated",
        required_flags=["_events_filtered"],
    )
    def calculate_impact_parameters(self):
        print(
            f"Calculating impact parameters for {self.name} with source {self.source}"
        )
        for split in ["all", "jet", "bkg"]:
            pf_data_dict = getattr(self, f"particle_data_{split}")
            if "d0" not in pf_data_dict:
                print(
                    f"No impact parameters found in {self.name} for split {split}. Skipping."
                )
                continue
            n_pf = np.array([len(el) for el in pf_data_dict["pt"]])
            n_pf_total = n_pf.sum()
            d0Sig = np.zeros(n_pf_total, dtype=float)
            z0Sig = np.zeros(n_pf_total, dtype=float)
            ip3d = np.zeros(n_pf_total, dtype=float)
            ip3dError = np.zeros(n_pf_total, dtype=float)
            ip3dSig = np.zeros(n_pf_total, dtype=float)
            curr_idx = 0
            for i in range(len(n_pf)):
                slice_ = slice(curr_idx, curr_idx + n_pf[i])
                d0Sig[slice_] = pf_data_dict["d0"][i] / (
                    pf_data_dict["d0Error"][i] + 1e-8
                )
                z0Sig[slice_] = pf_data_dict["z0"][i] / (
                    pf_data_dict["z0Error"][i] + 1e-8
                )
                ip3d[slice_] = np.sqrt(
                    pf_data_dict["d0"][i] ** 2 + pf_data_dict["z0"][i] ** 2
                )
                ip3dError[slice_] = np.sqrt(
                    pf_data_dict["d0Error"][i] ** 2 + pf_data_dict["z0Error"][i] ** 2
                )
                ip3dSig[slice_] = ip3d[slice_] / (ip3dError[slice_] + 1e-8)
                curr_idx += n_pf[i]
            pf_data_dict["d0Sig"] = np.array(
                np.split(d0Sig, np.cumsum(n_pf)[:-1]), dtype=object
            )
            pf_data_dict["z0Sig"] = np.array(
                np.split(z0Sig, np.cumsum(n_pf)[:-1]), dtype=object
            )
            pf_data_dict["ip3d"] = np.array(
                np.split(ip3d, np.cumsum(n_pf)[:-1]), dtype=object
            )
            pf_data_dict["ip3dError"] = np.array(
                np.split(ip3dError, np.cumsum(n_pf)[:-1]), dtype=object
            )
            pf_data_dict["ip3dSig"] = np.array(
                np.split(ip3dSig, np.cumsum(n_pf)[:-1]), dtype=object
            )

    def __repr__(self):
        return f"Data(name={self.name}, label={self.label}) with {len(self.event_number)} events."

    def get_vars(self):
        return {
            "event": list(self.event_data.keys()),
            "jet": list(self.jet_data.keys()),
            "particle_all": list(self.particle_data_all.keys()),
            "particle_jet": list(self.particle_data_jet.keys()),
            "particle_bkg": list(self.particle_data_bkg.keys()),
        }

    def print_vars(self):
        vars_ = self.get_vars()
        print(f"Data {self.name} variables:")
        for key, value in vars_.items():
            print(f"{key}: {value}")


@dataclass
class SampleData(Generic[SampleT]):
    sample: SampleT
    data_dict: dict[str, Data]

    # flags
    check_flags: bool = field(
        default=True, init=False
    )  # Check flags before running methods

    _events_filtered: bool = field(default=False, init=False)

    @classmethod
    def load(cls, sample: SampleT, config_list: list[PflowDataConfig]) -> Self:
        data_dict = {}
        cms_loaded = False
        for pflow_data_config in config_list:
            print(
                f"Loading data for {pflow_data_config.name} from {pflow_data_config.path}"
            )
            data_ = load_data(
                pflow_data_config.path,
                load_cms=not cms_loaded,
                entry_stop=pflow_data_config.num_events,
            )
            zero_neutral_vtx(data_)
            if not cms_loaded:
                pf_data = Data.load_from_dict(
                    name=Source.PF.name,
                    label="CMS PFlow",
                    source=Source.PF,
                    data_dict=data_["pflow"],
                    event_number=data_["eventNumber"],
                )
                tr_data = Data.load_from_dict(
                    name=Source.TR.name,
                    label="Truth",
                    source=Source.TR,
                    data_dict=data_["truth"],
                    event_number=data_["eventNumber"],
                )
                data_dict[pf_data.name] = pf_data
                data_dict[tr_data.name] = tr_data
                cms_loaded = True
            data_dict[pflow_data_config.name] = Data.load_from_dict(
                name=pflow_data_config.name,
                label=pflow_data_config.label,
                source=pflow_data_config.pflow_type,
                data_dict=data_["fastsim"],
                event_number=data_["eventNumber"],
            )
        return cls(sample=sample, data_dict=data_dict)

    @run_once("_events_filtered", "events already filtered")
    def filter_events(self):
        goodEvents = np.array(
            list(
                set.intersection(
                    *[
                        set(data.event_number.astype(int))
                        for data in self.data_dict.values()
                    ]
                )
            ),
            dtype=int,
        )
        if not goodEvents.size:
            raise ValueError("No common events found across all data sources.")
        for data in self.data_dict.values():
            data.filter_events(goodEvents)

    def disable_flags(self):
        self.check_flags = False
        for data in self.data_dict.values():
            data.check_flags = False

    def process_data(self):
        for data in self.data_dict.values():
            data.calculate_event_data()
        for data in self.data_dict.values():
            if data.source != Source.TR:
                print(f"Processing data for {data.name} with source {data.source}")
                data.calculate_event_residuals(self.data_dict[Source.TR.name])
                data.match_jets(self.data_dict[Source.TR.name])
                data.calculate_jet_residuals(self.data_dict[Source.TR.name])
                data.calculate_particle_residuals(self.data_dict[Source.TR.name])
                data.calculate_impact_parameters()
        print(f"Processed data for sample {self.sample.name}")


class Performance(Generic[SampleT]):
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.sample_data_dict: dict[SampleT, SampleData[SampleT]] = {}

    def load(self):
        for sample, config_list in self.config.sample_dict.items():
            self.sample_data_dict[sample] = SampleData[SampleT].load(
                sample, config_list
            )

    def process(self):
        for sample_data in self.sample_data_dict.values():
            print(f"Processing sample: {sample_data.sample.name}")
            sample_data.filter_events()
            sample_data.process_data()

    def disable_flags(self):
        for sample_data in self.sample_data_dict.values():
            sample_data.disable_flags()
