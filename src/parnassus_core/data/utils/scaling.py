import numpy as np
import torch
from collections import defaultdict

from sklearn.preprocessing import QuantileTransformer
import pickle

from parnassus_core.data.utils.variable_helper import VariableHelper
from parnassus_core.utils.pid import (
    N_PID_CLASSES,
    onehot_index_to_pdgid,
    pdgid_to_class,
    pdgid_to_onehot_index,
)

TRANSFORM_TYPES = {
    "std",
    "min_max",
    "min_max_sym",
    "quantile",
}

TRANSFORM_FUNCTIONS = {
    "log",
    "log1p",
    "sqrt",
    "tanh",
    "asinh",
    "atan",
    "pow",
    None,  # No function
}

# Relative momentum variables, unscaled back to the absolute variable by multiplying
# with the collection's event-level sum: base name -> (absolute name, sum name).
RELATIVE_VARS = {
    "ptrel": ("pt", "ht"),
}

# Per-particle transform blocks read with class_based_scaling: block-name prefix ->
# the per-particle field whose value selects the block (class_3, pdgId_-211, ...).
GROUP_BLOCK_PREFIXES = {
    "class_": "class",
    "pdgId_": "pdgId",
}


class VarTransform:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.type = self.config.get("type", "std")
        self.fn = self.config.get("fn", None)
        self.fn_scale = float(self.config.get("fn_scale", 1.0))
        self.fn_shift = float(self.config.get("fn_shift", 0.0))
        self.power = float(self.config.get("power", 1.0))
        self.clip = self.config.get("clip", None)
        self.qt_path = self.config.get("qt_path", None)

        if self.clip is not None:
            self.min_clip = self.clip.get("min", None)
            self.max_clip = self.clip.get("max", None)

        assert self.fn in TRANSFORM_FUNCTIONS, (
            f"Unknown transformation function {self.fn}. "
            f"Available functions: {TRANSFORM_FUNCTIONS}"
        )
        assert self.type in TRANSFORM_TYPES, (
            f"Unknown transformation type {self.type}. "
            f"Available types: {TRANSFORM_TYPES}"
        )

        if self.type == "std":
            self.shift = self.config.get("mean", 0.0)
            self.scale = self.config.get("std", 1.0)
        elif self.type == "min_max":
            self.shift = self.config.get("min", 0.0)
            self.scale = self.config.get("max", 1.0) - self.shift
        elif self.type == "min_max_sym":
            min_ = self.config.get("min", 0.0)
            max_ = self.config.get("max", 1.0)
            self.shift = (max_ + min_) / 2
            self.scale = (max_ - min_) / 2
        elif self.type == "quantile":
            self.shift = self.config.get("mean", 0.0)
            self.scale = self.config.get("std", 1.0)
            assert self.qt_path is not None, (
                "QuantileTransformer path must be provided."
            )
            with open(self.qt_path, "rb") as f:
                self.qt: QuantileTransformer = pickle.load(f)

    def transform(self, x, shift=None, scale=None):
        if self.clip is not None:
            x = torch.clamp(x, min=self.min_clip, max=self.max_clip)
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = (x + self.fn_shift) / self.fn_scale
        if self.type == "quantile":
            device = x.device
            shape = x.shape
            x_np = x.cpu().numpy().reshape(-1, 1)
            x_np = self.qt.transform(x_np)
            x = torch.from_numpy(x_np.reshape(shape)).to(device)
        match self.fn:
            case "log":
                # Added a clamp to avoid log blowing up to inf
                x = torch.log(torch.clamp(x, min=torch.finfo(x.dtype).tiny))
            case "log1p" | "sqrt" | "tanh" | "asinh" | "atan":
                x = getattr(torch, self.fn)(x)
            case "pow":
                x = torch.sign(x) * torch.pow(torch.abs(x), self.power)
            case "log-mod":
                x = torch.sign(x) * torch.log1p(torch.abs(x))
            case _:
                pass
        return (x - shift) / scale

    def inverse_transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
        if self.type == "quantile":
            device = x.device
            shape = x.shape
            x_np = x.cpu().numpy().reshape(-1, 1)
            x_np = self.qt.inverse_transform(x_np)
            x = torch.from_numpy(x_np.reshape(shape)).to(device)
        match self.fn:
            case "log":
                x = torch.exp(x)
            case "log1p":
                x = torch.expm1(x)
            case "sqrt":
                x = torch.pow(x, 2)
            case "tanh":
                x = torch.atanh(x)
            case "asinh":
                x = torch.sinh(x)
            case "atan":
                x = torch.tan(x)
            case "pow":
                x = torch.sign(x) * torch.pow(torch.abs(x), 1 / self.power)
            case "log-mod":
                x = torch.sign(x) * torch.expm1(torch.abs(x))
            case _:
                # For other functions, we assume they are linear transformations
                # and do not need to be inverted.
                pass
        x = x * self.fn_scale - self.fn_shift
        return x


def get_empty_transform() -> VarTransform:
    """
    Get an empty VarTransform that does not apply any transformation.

    Returns
    -------
    VarTransform
        An instance of VarTransform with no transformation applied.
    """
    return VarTransform("", {"type": "std", "mean": 0.0, "std": 1.0})


class FeatureScaler:
    def __init__(self, scale_dict: str, class_based_scaling: bool = False):
        """
        Initialize the FeatureScaler with a path to a YAML file containing scaling parameters.

        Parameters
        ----------
        scale_dict : str
            Path to the YAML file containing scaling parameters for features.
        class_based_scaling : bool
            Scale per-particle variables with per-group transform blocks: either
            ``class_<id>`` blocks, selected by the particle's ``class``, or
            ``pdgId_<id>`` blocks, selected by its ``pdgId`` bucketed like the
            pdgId one-hot (species without a bucket -> ``pdgId_0``). The two
            kinds cannot be mixed.
        """
        import yaml

        with open(scale_dict, "r") as f:
            self.scale_dict = yaml.safe_load(f)

        self.class_based_scaling = class_based_scaling
        # Per-particle field selecting the transform block: "class" or "pdgId".
        self.group_var: str | None = None
        self.class_transforms: dict[int, dict[str, VarTransform]] = {}
        self._class_var_transforms: dict[str, list[tuple[int, VarTransform]]] = (
            defaultdict(list)
        )
        self.transforms = defaultdict(get_empty_transform)
        for name, config in self.scale_dict.items():
            prefix = next((p for p in GROUP_BLOCK_PREFIXES if name.startswith(p)), None)
            if prefix is not None and self.class_based_scaling:
                group_var = GROUP_BLOCK_PREFIXES[prefix]
                if self.group_var not in (None, group_var):
                    raise ValueError(
                        "class_N and pdgId_N transform blocks cannot be mixed."
                    )
                self.group_var = group_var
                class_id = int(name[len(prefix) :])
                if not isinstance(config, dict):
                    raise ValueError(
                        f"Class config '{name}' must be a mapping of variable transforms."
                    )
                self.class_transforms[class_id] = defaultdict(get_empty_transform)
                for var_name, var_cfg in config.items():
                    tr = VarTransform(f"{name}.{var_name}", var_cfg)
                    self.class_transforms[class_id][var_name] = tr
                    self._class_var_transforms[var_name].append((class_id, tr))
                continue

            self.transforms[name] = VarTransform(name, config)

        if self.class_based_scaling and not self.class_transforms:
            raise ValueError(
                "class_based_scaling is enabled but no class_N or pdgId_N transform "
                "blocks were found."
            )

    def particle_groups(self, data: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Per-particle ids selecting the class_N / pdgId_N transform blocks.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            One collection's ``{bare_var: Tensor(…, N)}``.

        Returns
        -------
        torch.Tensor | None
            The ``class`` field, or with ``pdgId_N`` blocks the bucketed
            ``pdgId`` (the value ``Unscaler`` decodes from the one-hot, so both
            directions pick the same block); None if the field is absent.
        """
        if self.group_var == "pdgId":
            pdgid = data.get("pdgId")
            if pdgid is None:
                return None
            return onehot_index_to_pdgid(pdgid_to_onehot_index(pdgid))
        return data.get("class")

    def __getitem__(self, name: str) -> VarTransform:
        """
        Get the VarTransform for a specific feature name.

        Parameters
        ----------
        name : str
            The name of the feature.

        Returns
        -------
        VarTransform
            The VarTransform object for the specified feature.
        """
        return self.transforms[name]

    def _apply_class_transforms(
        self,
        name: str,
        x: torch.Tensor,
        particle_class: torch.Tensor | None,
        particle_class_masks: dict[int, torch.Tensor] | None,
        particle_class_indices: dict[int, tuple[torch.Tensor, ...]] | None,
        inverse: bool,
    ) -> torch.Tensor:
        if not self.class_based_scaling or particle_class is None:
            transform = self[name]
            return transform.inverse_transform(x) if inverse else transform.transform(x)

        class_items = self._class_var_transforms.get(name)
        if not class_items:
            # Identity fallback for variables without class-specific transforms.
            return x

        class_tensor = None
        index_lookup = particle_class_indices
        if index_lookup is None:
            index_lookup = particle_class_masks
        if index_lookup is None:
            class_tensor = particle_class.to(device=x.device)
            if class_tensor.ndim == x.ndim - 1:
                class_tensor = class_tensor.unsqueeze(-1)

        out = None
        for class_id, tr in class_items:
            indexer = None
            if index_lookup is not None and class_id in index_lookup:
                indexer = index_lookup[class_id]
            else:
                assert class_tensor is not None
                indexer = class_tensor == class_id

            if isinstance(indexer, tuple):
                if not indexer or indexer[0].numel() == 0:
                    continue
            else:
                if indexer.ndim == x.ndim - 1:
                    indexer = indexer.unsqueeze(-1)
                if not indexer.any():
                    continue

            if out is None:
                out = x.clone()
            selected = x[indexer]
            out[indexer] = (
                tr.inverse_transform(selected) if inverse else tr.transform(selected)
            )

        return out if out is not None else x

    def transform(
        self,
        name: str,
        x: torch.Tensor,
        particle_class: torch.Tensor | None = None,
        particle_class_masks: dict[int, torch.Tensor] | None = None,
        particle_class_indices: dict[int, tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        """Apply forward transform, optionally conditioned on per-particle class IDs."""
        return self._apply_class_transforms(
            name,
            x,
            particle_class,
            particle_class_masks,
            particle_class_indices,
            inverse=False,
        )

    def inverse_transform(
        self,
        name: str,
        x: torch.Tensor,
        particle_class: torch.Tensor | None = None,
        particle_class_masks: dict[int, torch.Tensor] | None = None,
        particle_class_indices: dict[int, tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        """Apply inverse transform, optionally conditioned on per-particle class IDs."""
        return self._apply_class_transforms(
            name,
            x,
            particle_class,
            particle_class_masks,
            particle_class_indices,
            inverse=True,
        )


class Unscaler:
    def __init__(
        self,
        feature_scaler: FeatureScaler,
        fs_vars: list[str],
        ctxt_vars: list[str],
        ctxt_global_vars: list[str],
        sin_cos: bool = True,
    ):
        """
        Initialize the Unscaler with a FeatureScaler and lists of feature names.

        Parameters
        ----------
        feature_scaler : FeatureScaler
            The FeatureScaler instance used for scaling transformations.
        fs_vars : list[str]
            List of fast simulation feature names.
        ctxt_vars : list[str]
            List of context feature names.
        ctxt_global_vars : list[str]
            List of global context feature names.
        sin_cos : bool, optional
            Whether to have sine and cosine transformations for angular variables, by default True.
        """
        self.feature_scaler = feature_scaler
        self.fs_vars = fs_vars
        self.ctxt_vars = ctxt_vars
        self.ctxt_global_vars = ctxt_global_vars
        self.sin_cos = sin_cos

        self.var_helper = VariableHelper(
            [
                el.split("_")[0]
                for el in set(self.fs_vars + self.ctxt_vars + self.ctxt_global_vars)
                if "_" in el
            ]
        )

        self.primary_coll = self.var_helper.get_collection(self.ctxt_vars[0])

        self.mean_vars = []
        for var in self.ctxt_vars:
            collection = self.var_helper.get_collection(var)
            base_name = self.var_helper.get_base_name(var)
            if (
                collection == self.primary_coll
                and ("charge" not in base_name)
                and ("class" not in base_name)
                and ("pdgId" not in base_name)
            ):
                self.mean_vars.append(base_name)

        target_collection = self.var_helper.get_collection(self.fs_vars[0])
        ctxt_collection = self.var_helper.get_collection(self.ctxt_vars[0])

        assert target_collection is not None, (
            "Target collection could not be inferred from fs_vars."
        )
        assert ctxt_collection is not None, (
            "Context collection could not be inferred from ctxt_vars."
        )
        self.target_collection = target_collection
        self.ctxt_collection = ctxt_collection

        # Unshifting adds the primary vertex z back to vz_shifted variables
        # only, so pv_coords is needed only when one of them is present.
        self.has_vz_shifted = any(
            self.var_helper.get_base_name(var) == "vz_shifted"
            for var in self.fs_vars + self.ctxt_vars
        )

    def _unscale_per_particle_vars(
        self,
        data: torch.Tensor,
        vars_list: list[str],
        new_data_dict: dict,
        pv_z: np.ndarray | None,
        unshift: bool,
    ) -> torch.Tensor | None:
        """Unscale a per-particle tensor into new_data_dict using vars_list ordering.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor to unscale.
        vars_list : list[str]
            The list of variable names to unscale.
        new_data_dict : dict
            The dictionary to store the unscaled variables.
        pv_z : np.ndarray or None
            Per-event primary vertex z, shape (B, 1); None when no variable
            needs unshifting.
        unshift : bool
            Whether to unshift the variables.

        Returns
        -------
        particle_class : torch.Tensor
            Class IDs extracted from one-hot features, if present; otherwise None.
        """
        particle_class = self._extract_class_ids(data, vars_list)
        var_idx = 0
        for var_name in vars_list:
            if var_idx >= data.shape[-1]:
                break
            collection = self.var_helper.get_collection(var_name)
            assert collection is not None, (
                f"Context variable {var_name} must belong to a collection."
            )
            base_name = self.var_helper.get_base_name(var_name)
            scaler_key = self.var_helper.scaler_key(var_name)
            var, index_shift = self.extract_var(
                data,
                scaler_key,
                var_idx,
                self.sin_cos,
                particle_class=particle_class,
            )
            var_idx += index_shift
            var = var.numpy()
            if base_name in RELATIVE_VARS:
                base_name, sum_name = RELATIVE_VARS[base_name]
                var = var * new_data_dict[f"{collection}_{sum_name}"].reshape(-1, 1)
            if base_name == "vz_shifted" and unshift:
                var = var + pv_z
                base_name = "vz"
            new_data_dict[f"{collection}_{base_name}"] = var
        return particle_class

    def _extract_class_ids(
        self, data: torch.Tensor, vars_list: list[str]
    ) -> torch.Tensor | None:
        """Decode class IDs from one-hot class features for class-aware inverse scaling."""
        var_idx = 0
        for var_name in vars_list:
            if var_idx >= data.shape[-1]:
                break
            base_name = self.var_helper.get_base_name(var_name)
            scaler_key = self.var_helper.scaler_key(var_name)
            if scaler_key.endswith("phi") and self.sin_cos:
                var_idx += 2
            elif scaler_key.endswith("charge"):
                var_idx += 3
            elif scaler_key.endswith("class"):
                return data[..., var_idx : var_idx + 5].argmax(-1)
            elif scaler_key.endswith("pdgId"):
                # PID config: if there is no 'class' target, the decoded pdgId selects
                # the pdgId_N blocks directly, or its class (via pdgid_class_dict)
                # the class_N blocks.
                idx = data[..., var_idx : var_idx + N_PID_CLASSES].argmax(-1)
                pdgid = onehot_index_to_pdgid(idx)
                if self.feature_scaler.group_var == "pdgId":
                    return pdgid
                return pdgid_to_class(pdgid)
            else:
                var_idx += 1
            if base_name.endswith("class"):
                return data[..., var_idx - 5 : var_idx].argmax(-1)
        return None

    def extract_var(
        self,
        data: torch.Tensor,
        var_name: str,
        var_idx: int,
        sin_cos: bool = False,
        particle_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        index_shift = 0
        if var_name.endswith("phi") and sin_cos:
            var = torch.atan2(
                data[..., var_idx],
                data[..., var_idx + 1],
            )
            index_shift = 2
        elif var_name.endswith("charge"):
            var = data[..., var_idx : var_idx + 3].argmax(-1) - 1
            index_shift = 3
        elif var_name.endswith("class"):
            var = data[..., var_idx : var_idx + 5].argmax(-1)
            index_shift = 5
        elif var_name.endswith("pdgId"):
            idx = data[..., var_idx : var_idx + N_PID_CLASSES].argmax(-1)
            var = onehot_index_to_pdgid(idx)
            index_shift = N_PID_CLASSES
        else:
            var = self.feature_scaler.inverse_transform(
                var_name,
                data[..., var_idx],
                particle_class=particle_class,
            )
            index_shift = 1
        return var, index_shift

    def unscale_variables(
        self,
        fs_data: torch.Tensor,
        ctxt_dict: dict[str, torch.Tensor],
        target_data: torch.Tensor | None = None,
        unshift: bool = True,
    ) -> dict[str, np.ndarray]:
        """Unscale features from tensors and store in new_data_dict.

        Parameters
        ----------
        fs_data : torch.Tensor
            The fast simulation data tensor to unscale.
        ctxt_dict : dict[str, torch.Tensor]
            Dictionary containing context tensors to unscale.
        target_data : torch.Tensor, optional
            The target data tensor to unscale (if available), by default None.
        unshift : bool, optional
            Whether to unshift vz_shifted variables by adding back the primary
            vertex z, by default True. Requires ``ctxt_dict["pv_coords"]`` only
            when some fs or context variable is vz_shifted.

        Returns
        -------
        new_data_dict : dict[str, np.ndarray]
            A dictionary mapping variable names to their unscaled numpy arrays.
        """
        fs_data = fs_data.cpu()
        has_target = target_data is not None
        if has_target:
            target_data = target_data.cpu()
        ctxt_dict = {k: v.cpu() for k, v in ctxt_dict.items()}
        new_data_dict = {}

        # Global context variables
        var_idx = 0
        for var_name in self.ctxt_global_vars:
            collection = self.var_helper.get_collection(var_name)
            base_name = self.var_helper.get_base_name(var_name)
            scaler_key = self.var_helper.scaler_key(var_name)
            if var_name == "means":
                var_idx += len(self.mean_vars)
            elif collection in ("pflow", "truth"):
                new_data_dict[f"{collection}_{base_name}"] = (
                    self.feature_scaler[scaler_key]
                    .inverse_transform(ctxt_dict["ctxt_global_data"][..., var_idx])
                    .numpy()
                )
                var_idx += 1
            else:
                new_data_dict[f"ctxt_{base_name}"] = (
                    self.feature_scaler[scaler_key]
                    .inverse_transform(ctxt_dict["ctxt_global_data"][..., var_idx])
                    .numpy()
                )
                var_idx += 1

        pv_z = None
        if unshift and self.has_vz_shifted:
            if "pv_coords" not in ctxt_dict:
                raise KeyError(
                    "Unshifting vz_shifted needs ctxt_dict['pv_coords']; load the "
                    "dataset with recalculate_ip=True or pass unshift=False."
                )
            pv_z = ctxt_dict["pv_coords"][:, -1].numpy().reshape(-1, 1)

        # Per-particle context variables
        self._unscale_per_particle_vars(
            ctxt_dict["ctxt_data"], self.ctxt_vars, new_data_dict, pv_z, unshift
        )

        # Pflow-only context variables (separate tensor)
        pf_class = None
        if "pf_ctxt_data" in ctxt_dict:
            pflow_vars = [
                v
                for v in self.ctxt_vars
                if self.var_helper.get_collection(v) == "pflow"
            ]
            pf_class = self._unscale_per_particle_vars(
                ctxt_dict["pf_ctxt_data"], pflow_vars, new_data_dict, pv_z, unshift
            )

        # Fast-sim (generated) variables
        if pf_class is not None and self.target_collection == "pflow":
            fs_class = pf_class
            target_class = pf_class if has_target else None
        else:
            fs_class = self._extract_class_ids(fs_data, self.fs_vars)
            target_class = (
                self._extract_class_ids(target_data, self.fs_vars)
                if has_target
                else None
            )
        var_idx = 0
        for var_name in self.fs_vars:
            collection = self.var_helper.get_collection(var_name)
            assert collection is not None, (
                f"Feature variable {var_name} must belong to a collection."
            )
            base_name = self.var_helper.get_base_name(var_name)
            scaler_key = self.var_helper.scaler_key(var_name)

            fs_, index_shift = self.extract_var(
                fs_data,
                scaler_key,
                var_idx,
                self.sin_cos,
                particle_class=fs_class,
            )
            fs_ = fs_.numpy()

            target_ = None
            if has_target:
                target_, _ = self.extract_var(
                    target_data,
                    scaler_key,
                    var_idx,
                    self.sin_cos,
                    particle_class=target_class,
                )
                target_ = target_.numpy()

            var_idx += index_shift

            if base_name in RELATIVE_VARS:
                base_name, sum_name = RELATIVE_VARS[base_name]
                ht = new_data_dict[f"{collection}_{sum_name}"].reshape(-1, 1)
                fs_ = fs_ * ht
                target_ = target_ * ht if has_target else None
            if base_name == "vz_shifted" and unshift:
                fs_ = fs_ + pv_z
                if has_target and target_ is not None:
                    target_ = target_ + pv_z
                base_name = "vz"

            new_data_dict[f"fs_{base_name}"] = fs_
            if has_target:
                new_data_dict[f"{collection}_{base_name}"] = target_

        if "ctxt_mask" in ctxt_dict:
            new_data_dict["ctxt_mask"] = ctxt_dict["ctxt_mask"].numpy()
        if "fs_mask" in ctxt_dict:
            new_data_dict["fs_mask"] = ctxt_dict["fs_mask"].numpy()
        if "eventNumber" in ctxt_dict:
            new_data_dict["eventNumber"] = ctxt_dict["eventNumber"].numpy()
        return new_data_dict
