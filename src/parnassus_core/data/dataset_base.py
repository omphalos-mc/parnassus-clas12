"""Shared base class for physics datasets.

``PhysicsDatasetBase`` implements ``__getitem__`` once — all transform,
assembly, and output-key logic lives here.  Subclasses only override
``_get_collections()`` to produce data in the **common intermediate format**.

Subclasses
----------
``EagerDataset``  (ROOT, LUND)
    Loads everything at ``__init__``, stores per-collection ragged arrays +
    cumsum.  ``_get_collections(int)`` slices one event, sorts by p,
    pads to ``max_particles``.

``StreamingDataset``  (HDF5)
    Keeps ``HDF5FileLoader`` open.  ``_get_collections(slice)`` does
    ``read_direct`` per batch, returning pre-padded 2-D tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from torch import Tensor
from torch.utils.data import Dataset

from parnassus_core.data.physics_data import (
    build_ctxt_data,
    build_ctxt_global_data,
    build_fs_data,
    compute_means,
)
from parnassus_core.data.utils.variable_helper import VariableHelper, VarConfig
from parnassus_core.data.utils.scaling import FeatureScaler


class PhysicsDatasetBase(Dataset, ABC):
    """Abstract base with a fully shared ``__getitem__``.

    Subclasses **must** implement :meth:`_get_collections` and call
    ``super().__init__()`` with the common configuration parameters.

    Parameters (accepted by ``__init__``)
    --------------------------------------
    var_dict : str
        Path to YAML variable-list config.
    scale_dict : str
        Path to YAML scaling config.
    max_particles : int
        Padding length per collection.
    train_type : ``"particle"`` | ``"event"`` | ``"impact"``
    sin_cos : bool
        Apply sin/cos transform to phi variables.
    zero_neutral_vtx : bool
        Zero vertex-related variables for neutral particles.
    fs_mask_type : ``"all"`` | ``"charged"`` | ``"neutral"``
    ctxt_mask_type : ``"all"`` | ``"charged"`` | ``"neutral"``
    mode : ``"train"`` | ``"test"`` | ``"eval"``
    recalculate_ip : bool
        Recompute impact parameters from vertex positions.
    class_based_scaling : bool
        Whether to use class-aware scaling (requires one-hot class features).
    eta_clamp : tuple[float, float] | None
        Inclusive ``(min, max)`` bounds that ``eta`` is clamped to for every
        collection. Values outside the range saturate to the nearest bound
        (they are not dropped). ``None`` disables clamping entirely. Default
        ``(-3.0, 3.0)`` preserves the historical CMS acceptance.
    """

    # Variables zeroed for neutrals when zero_neutral_vtx is True.
    # NOTE: vz_shifted is intentionally excluded — legacy behaviour computes
    # vz_shifted from the *raw* vz and never zeros it for neutrals.
    _NEUTRAL_ZERO_VARS = frozenset(
        {
            "vx",
            "vy",
            "vz",
            "d0",
            "z0",
            "d0My",
            "z0My",
            "d0Error",
            "z0Error",
            "d0Sig",
            "z0Sig",
        }
    )

    def __init__(
        self,
        var_dict: str,
        scale_dict: str,
        max_particles: int,
        train_type: Literal["particle", "event", "impact"],
        sin_cos: bool,
        zero_neutral_vtx: bool,
        fs_mask_type: Literal["all", "charged", "neutral"] = "all",
        ctxt_mask_type: Literal["all", "charged", "neutral"] = "all",
        mode: Literal["train", "test", "eval"] = "train",
        recalculate_ip: bool = False,
        class_based_scaling: bool = False,
        eta_clamp: tuple[float, float] | None = (-3.0, 3.0),
    ):
        super().__init__()

        # ---- Store config ----
        self.feature_scaler = FeatureScaler(
            scale_dict, class_based_scaling=class_based_scaling
        )
        self.max_particles = max_particles
        self.train_type = train_type
        self.sin_cos = sin_cos
        self.zero_neutral_vtx = zero_neutral_vtx
        self.fs_mask_type = fs_mask_type
        self.ctxt_mask_type = ctxt_mask_type
        self.mode = mode
        self.recalculate_ip = recalculate_ip
        self.class_based_scaling = class_based_scaling
        self.eta_clamp = eta_clamp

        assert self.mode in ("train", "test", "eval"), (
            f"Mode must be 'train', 'test' or 'eval', got '{self.mode}'"
        )

        # ---- Variable config ----
        self._vcfg = VarConfig.from_yaml(var_dict)
        self.ctxt_vars: list[str] = list(self._vcfg.ctxt_vars)
        self.fs_vars = self._vcfg.fs_vars
        self.ctxt_global_vars: list[str] = list(self._vcfg.ctxt_global_vars)
        self.event_vars = self._vcfg.event_vars

        # ---- Collection setup (var_helper, eval filter) ----
        self._all_collection_names = list(self._vcfg.vars_to_load.keys())
        self.var_helper = VariableHelper(self._all_collection_names)

        if self.mode == "eval":
            self.ctxt_vars, self.ctxt_global_vars, self._collections_to_load = (
                self._apply_eval_filter(
                    self.ctxt_vars,
                    self.ctxt_global_vars,
                    self._vcfg.vars_to_load,
                    self.var_helper,
                )
            )
        else:
            self._collections_to_load = dict(self._vcfg.vars_to_load)

    # ------------------------------------------------------------------
    # Shared init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_event_range(
        total_events: int,
        reduce_ds: float | int,
        entry_start: int,
    ) -> int:
        """Compute number of events to load from *entry_start*.

        Parameters
        ----------
        total_events : int
            Total events in file.
        reduce_ds : float | int
            ``<1.0`` → fraction of available events; ``>=1.0`` → count.
        entry_start : int
            First event index.

        Returns
        -------
        int
            Number of events to load.
        """
        available = total_events - entry_start
        if 0 < reduce_ds < 1.0:
            return int(available * reduce_ds)
        if reduce_ds >= 1.0:
            return min(int(reduce_ds), available)
        return available

    # ------------------------------------------------------------------
    # Config parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_eval_filter(
        ctxt_vars: list[str],
        ctxt_global_vars: list[str],
        vars_to_load: dict[str, list[str]],
        var_helper: VariableHelper,
    ) -> tuple[list[str], list[str], dict[str, list[str]]]:
        """Filter collections and variable lists for eval mode.

        Keeps only collections referenced by *ctxt_vars*, then prunes
        variable lists to only reference loaded collections.

        Returns ``(ctxt_vars, ctxt_global_vars, collections_to_load)``.
        """
        primary_colls = {
            var_helper.get_collection(v)
            for v in ctxt_vars
            if var_helper.get_collection(v) is not None
        }
        collections = {c: vs for c, vs in vars_to_load.items() if c in primary_colls}
        loaded = set(collections.keys())
        ctxt_vars = [
            v
            for v in ctxt_vars
            if var_helper.get_collection(v) is None
            or var_helper.get_collection(v) in loaded
        ]
        ctxt_global_vars = [
            v
            for v in ctxt_global_vars
            if (
                var_helper.get_collection(v) is None
                or var_helper.get_collection(v) in loaded
            )
            and (
                var_helper.is_npart_var(v) is None
                or var_helper.is_npart_var(v) in loaded
            )
        ]
        return ctxt_vars, ctxt_global_vars, collections

    # ------------------------------------------------------------------
    # Abstract — sole polymorphic method
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_collections(
        self,
        idx: int | slice,
    ) -> tuple[
        dict[str, dict[str, Tensor]],  # collection_data
        dict[str, Tensor],  # collection_masks
        dict[str, Tensor],  # event_data
        Tensor | None,  # pre-computed means (or None)
    ]:
        """Produce data in the common intermediate format.

        EagerDataset (ROOT, LUND)
            ``idx`` is ``int``.  Cumsum-slice one event from per-collection
            ragged arrays, sort by p descending, pad to
            ``max_particles``.  Returns pre-computed means.

        StreamingDataset (HDF5)
            ``idx`` is ``slice``.  ``read_direct`` a batch from h5py,
            convert to tensors.  Returns ``means=None`` (computed in
            ``__getitem__`` via ``compute_means``).

        Returns
        -------
        collection_data : dict[str, dict[str, Tensor]]
            ``{collection: {bare_var: Tensor(…, N)}}``
        collection_masks : dict[str, Tensor]
            ``{collection: BoolTensor(…, N)}``
        event_data : dict[str, Tensor]
            ``{var_name: Tensor(…,)}``
        means : Tensor | None
            Pre-computed ``(…, n_mean_vars)`` or ``None``.
        """
        ...

    # ------------------------------------------------------------------
    # Shared __getitem__
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int | slice
    ) -> tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]:
        """Build a training/eval sample from the common intermediate format.

        1. Calls ``_get_collections(idx)`` (subclass).
        2. Optionally zeros vertex vars for neutrals.
        3. Builds context data (``ctxt_data``, ``pf_ctxt_data``).
        4. Computes or retrieves means.
        5. Builds global context (means prepended).
        6. In train/test: builds target data (``fs_data``, ``fs_mask``).

        Returns
        -------
        train / test mode
            ``(fs_data, ctxt_dict)``
        eval mode
            ``ctxt_dict``
        """
        collection_data, collection_masks, event_data, means = self._get_collections(
            idx
        )

        # ---- zero neutral vertex vars ----
        if self.zero_neutral_vtx:
            self._zero_neutral_vtx(collection_data, collection_masks)

        # ---- context ----
        ctxt_dict = build_ctxt_data(
            self.ctxt_vars,
            collection_data,
            collection_masks,
            self.sin_cos,
            self.feature_scaler,
            self.var_helper,
        )

        # ---- means ----
        if means is None and "means" in self.ctxt_global_vars:
            primary_coll = self.var_helper.get_collection(self.ctxt_vars[0])
            if primary_coll is not None and primary_coll in collection_data:
                means = compute_means(
                    collection_data[primary_coll],
                    self.ctxt_vars,
                    collection_masks[primary_coll],
                    self.var_helper,
                    primary_coll,
                )

        # ---- global context ----
        ctxt_dict["ctxt_global_data"] = build_ctxt_global_data(
            self.ctxt_global_vars,
            collection_masks,
            event_data,
            self.feature_scaler,
            self.var_helper,
            means,
        ).nan_to_num(0.0)

        # ---- event number (metadata) ----
        if "eventNumber" in event_data:
            ctxt_dict["eventNumber"] = event_data["eventNumber"]

        # ---- pv_coords (when recalculate_ip is used) ----
        if "pv_coords" in event_data:
            ctxt_dict["pv_coords"] = event_data["pv_coords"]

        # ---- target (train / test only) ----
        if self.mode != "eval":
            fs_data, fs_mask = build_fs_data(
                self.fs_vars,
                collection_data,
                collection_masks,
                event_data,
                self.sin_cos,
                self.feature_scaler,
                self.var_helper,
                self.train_type,  # type: ignore
                self.fs_mask_type,  # type: ignore
            )
            fs_data = fs_data.nan_to_num(0.0)
            ctxt_dict["fs_mask"] = fs_mask
            return fs_data, ctxt_dict

        return ctxt_dict

    # ------------------------------------------------------------------
    # Neutral-vertex zeroing
    # ------------------------------------------------------------------

    _SENTINEL_CLEAN_VARS = frozenset(
        {"d0", "z0", "d0Error", "z0Error", "d0Sig", "z0Sig"}
    )

    def _zero_neutral_vtx(
        self,
        collection_data: dict[str, dict[str, Tensor]],
        collection_masks: dict[str, Tensor],
    ) -> None:
        """In-place zero vertex vars for neutrals (pflow only, not truth).

        Also cleans sentinel values ``< -998 → 0`` on d0/z0-related
        variables for all particles in affected collections.
        """
        ref_coll = self.var_helper.collection_names[0]
        for coll_name, data in collection_data.items():
            if coll_name == ref_coll or "class" not in data:
                continue
            neutral = data["class"] > 2
            for v in self._NEUTRAL_ZERO_VARS:
                if v in data:
                    data[v] = data[v].masked_fill(neutral, 0.0)
            for v in self._SENTINEL_CLEAN_VARS:
                if v in data:
                    data[v] = data[v].masked_fill(data[v] < -998, 0.0)
