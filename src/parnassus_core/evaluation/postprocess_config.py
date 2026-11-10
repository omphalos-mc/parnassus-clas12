"""Configuration layer for the post-processing step (``postprocess_eval.py``).

Mirrors the ``evaluation/config.py`` pattern used by ``eval.py``: a dataclass
whose field defaults reproduce the historical CMS/collider behavior exactly, so
running without ``-c/--config`` is unchanged. A YAML config (typically
per-experiment, e.g. CLAS12_RGA) overrides variable lists, branch layout,
acceptance cuts, and jet settings *without editing the module*.

Precedence: with ``-c`` the YAML is the source of truth for physics/vars; the
run-scoped CLI flags (``-i``, ``-d``, ``-o``, ``-n``, ``-es``) stay on the CLI.
Without ``-c``, :meth:`PostProcessConfig.from_cli` maps the existing physics
flags onto the defaults so legacy invocations behave identically.
"""

from typing import Self
from pathlib import Path
from dataclasses import dataclass, field, fields

import yaml


@dataclass
class PostProcessConfig:
    # --- variable lists (primary override targets) -------------------------
    particle_vars: list[str] = field(
        default_factory=lambda: ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
    )
    particle_ip_vars: list[str] = field(
        default_factory=lambda: ["d0", "d0Error", "z0", "z0Error"]
    )
    jet_vars: list[str] = field(
        default_factory=lambda: ["pt", "eta", "phi", "d2", "c2"]
    )
    event_vars: list[str] = field(default_factory=lambda: ["eventNumber"])
    matching_vars: list[str] = field(default_factory=lambda: ["tr_idx", "pf_idx"])

    # --- reader / branch layout --------------------------------------------
    # Branch layout is detected per input file when "auto" (eval output is
    # nested "truth.pt"; reference data is often flat "truth_pt"), or forced.
    branch_layout: str = "auto"  # "auto" | "nested" | "flat"

    # --- event join ---------------------------------------------------------
    # How the -i (fastsim) and -d (reference) event sets are combined.
    #   "intersect" -- only events present in both (the historical behavior).
    #   "union"     -- every event in either file; the collection a file does not
    #                  cover is written empty for that event. Use when the two
    #                  sides are deliberately different selections and neither
    #                  should be filtered by the other (e.g. all classifier-kept
    #                  fastsim events vs all truly-reconstructed reference events).
    join: str = "intersect"  # "intersect" | "union"
    # Drop joined events above this eventNumber. Needed with "union" when the
    # reference file extends past the last event the eval run consumed --
    # those rows have no fastsim counterpart by construction. Cannot be expressed
    # with -n, which is shared by both load_file() calls and also caps the number
    # of processed events.
    max_event_number: int | None = None

    # --- acceptance cuts (see step 2) --------------------------------------
    pt_min: float = 1.0
    pt_max: float = 1e4
    eta_cut: float = 3.0

    # --- jet clustering (see step 3) ---------------------------------------
    cluster_jets: bool = True
    jet_algorithm: str = "antikt"
    jet_radius: float = 0.5
    jet_ptmin: float = 20.0
    jet_min_constituents: int = 2
    calc_substructure: bool = True

    # --- existing flag behavior --------------------------------------------
    ip: bool = False
    matching: bool = False
    dr_cut: float = 0.6
    df: bool = False
    dl: bool = False

    # --- per-particle four-vector output -----------------------------------
    # When True, write derived px/py/pz/E branches for each particle collection
    # (truth/pflow/fastsim) regardless of whether jets are clustered. E respects
    # refine_energy (pdgId mass). Default False keeps CMS output unchanged.
    fourvec: bool = False

    # --- later steps (energy refinement / DIS) -----------------------------
    refine_energy: bool = False
    compute_dis: bool = False
    beam_energy: float = 10.6
    dis_vars: list[str] = field(default_factory=lambda: ["Q2", "xB", "W", "y", "nu"])
    target_mass: float = 0.93827

    def __post_init__(self):
        if self.branch_layout not in ("auto", "nested", "flat"):
            raise ValueError(
                f"branch_layout must be 'auto'/'nested'/'flat', got {self.branch_layout!r}"
            )
        if self.join not in ("intersect", "union"):
            raise ValueError(f"join must be 'intersect'/'union', got {self.join!r}")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Self:
        yaml_path = Path(yaml_path)
        assert yaml_path.exists(), f"Config file {yaml_path} does not exist."
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown key(s) in {yaml_path}: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}."
            )
        return cls(**data)

    @classmethod
    def from_cli(cls, args) -> Self:
        """Build a default config, mapping the legacy physics CLI flags onto it
        so a no-``-c`` run reproduces historical behavior."""
        return cls(
            eta_cut=args.eta,
            jet_radius=args.dr,
            ip=args.ip,
            matching=args.matching,
            dr_cut=args.dr_cut,
            df=args.df,
            dl=args.dl,
            join=getattr(args, "join", "intersect"),
            max_event_number=getattr(args, "max_event_number", None),
        )
