from dataclasses import dataclass
from typing import Any, Optional

import yaml


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config '{name}' must be a mapping")
    return value


def _as_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
        raise ValueError(f"Config '{name}' must be a list of integers")
    return value


def _as_optional_number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config '{name}' must be a number")
    return float(value)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Config '{name}' must be a boolean")
    return value


@dataclass
class VariableConfig:
    """Configuration for a single variable, including optional transform settings."""

    fn: Optional[str] = None
    fn_scale: Optional[float] = None
    fn_shift: float = 0.0
    deferred: bool = False
    charged: bool = False
    common: bool = False
    abs_clip: Optional[float] = None

    def __post_init__(self):
        if self.fn is not None and not isinstance(self.fn, str):
            raise ValueError("Variable field 'fn' must be a string when provided")
        self.fn_scale = _as_optional_number(self.fn_scale, "*.fn_scale")
        shift = _as_optional_number(self.fn_shift, "*.fn_shift")
        self.fn_shift = 0.0 if shift is None else shift
        self.abs_clip = _as_optional_number(self.abs_clip, "*.abs_clip")
        self.deferred = _as_bool(self.deferred, "*.deferred")
        self.charged = _as_bool(self.charged, "*.charged")
        self.common = _as_bool(self.common, "*.common")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariableConfig":
        return cls(
            fn=data.get("fn"),
            fn_scale=data.get("fn_scale"),
            fn_shift=data.get("fn_shift", 0.0),
            deferred=data.get("deferred", False),
            charged=data.get("charged", False),
            common=data.get("common", False),
            abs_clip=data.get("abs_clip"),
        )


@dataclass
class RuntimeConfig:
    event_vars: dict[str, VariableConfig]
    particle_vars: dict[str, VariableConfig]
    class_ids: list[int]
    charged_class_lt: int
    recalculate_pv: bool

    def __post_init__(self):
        if not self.event_vars:
            raise ValueError("Config 'event_vars' cannot be empty")
        if not self.particle_vars:
            raise ValueError("Config 'particle_vars' cannot be empty")
        self.class_ids = _as_int_list(self.class_ids, "settings.class_ids")
        if not isinstance(self.charged_class_lt, int):
            raise ValueError("Config 'settings.charged_class_lt' must be an integer")
        self.recalculate_pv = _as_bool(self.recalculate_pv, "settings.recalculate_pv")

        # If shifted-z is configured, PV recalculation is always required.
        if "vz_shifted" in self.particle_vars:
            self.recalculate_pv = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        root = _as_mapping(data, "root")
        event_raw = _as_mapping(root.get("event_vars"), "event_vars")
        particle_raw = _as_mapping(root.get("particle_vars"), "particle_vars")

        event_vars = {
            name: VariableConfig.from_dict(
                _as_mapping(cfg if cfg is not None else {}, f"event_vars.{name}")
            )
            for name, cfg in event_raw.items()
        }
        particle_vars = {
            name: VariableConfig.from_dict(
                _as_mapping(cfg if cfg is not None else {}, f"particle_vars.{name}")
            )
            for name, cfg in particle_raw.items()
        }

        settings = _as_mapping(root.get("settings", {}), "settings")
        class_ids = settings.get("class_ids", root.get("class_ids", [0, 1, 2, 3, 4]))
        charged_class_lt = settings.get(
            "charged_class_lt", root.get("charged_class_lt", 3)
        )
        recalculate_pv = settings.get(
            "recalculate_pv", root.get("recalculate_pv", False)
        )

        return cls(
            event_vars=event_vars,
            particle_vars=particle_vars,
            class_ids=class_ids,
            charged_class_lt=charged_class_lt,
            recalculate_pv=recalculate_pv,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
