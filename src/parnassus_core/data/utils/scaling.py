import torch
from collections import defaultdict

from sklearn.preprocessing import QuantileTransformer
import pickle

from parnassus_core.data.utils.variable_helper import VariableHelper

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
            case "log" | "log1p" | "sqrt" | "tanh" | "asinh" | "atan":
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
    def __init__(self, scale_dict: str):
        """
        Initialize the FeatureScaler with a path to a YAML file containing scaling parameters.

        Parameters
        ----------
        scale_dict : str
            Path to the YAML file containing scaling parameters for features.
        """
        import yaml

        with open(scale_dict, "r") as f:
            self.scale_dict = yaml.safe_load(f)
        self.transforms = defaultdict(get_empty_transform)
        for name, config in self.scale_dict.items():
            self.transforms[name] = VarTransform(name, config)

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

    def _unscale_per_particle_vars(self, data, vars_list, new_data_dict, pv_z, unshift):
        """Unscale a per-particle tensor into new_data_dict using vars_list ordering."""
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
            var, index_shift = self.extract_var(data, scaler_key, var_idx, self.sin_cos)
            var_idx += index_shift
            var = var.numpy()
            if base_name == "ptrel":
                var = var * new_data_dict[f"{collection}_ht"].reshape(-1, 1)
                base_name = "pt"
            if base_name == "vz_shifted" and unshift:
                var = var + pv_z
                base_name = "vz"
            new_data_dict[f"{collection}_{base_name}"] = var

    def extract_var(self, data, var_name, var_idx, sin_cos=False):
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
        else:
            var = self.feature_scaler[var_name].inverse_transform(data[..., var_idx])
            index_shift = 1
        return var, index_shift

    def unscale_variables(self, fs_data, ctxt_dict, target_data=None, unshift=True):
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

        pv_z = ctxt_dict["pv_coords"][:, -1].numpy().reshape(-1, 1) if unshift else None

        # Per-particle context variables
        self._unscale_per_particle_vars(
            ctxt_dict["ctxt_data"], self.ctxt_vars, new_data_dict, pv_z, unshift
        )

        # Pflow-only context variables (separate tensor)
        if "pf_ctxt_data" in ctxt_dict:
            pflow_vars = [
                v
                for v in self.ctxt_vars
                if self.var_helper.get_collection(v) == "pflow"
            ]
            self._unscale_per_particle_vars(
                ctxt_dict["pf_ctxt_data"], pflow_vars, new_data_dict, pv_z, unshift
            )

        # Fast-sim (generated) variables
        var_idx = 0
        for var_name in self.fs_vars:
            collection = self.var_helper.get_collection(var_name)
            assert collection is not None, (
                f"Feature variable {var_name} must belong to a collection."
            )
            base_name = self.var_helper.get_base_name(var_name)
            scaler_key = self.var_helper.scaler_key(var_name)

            fs_, index_shift = self.extract_var(
                fs_data, scaler_key, var_idx, self.sin_cos
            )
            fs_ = fs_.numpy()

            target_ = None
            if has_target:
                target_, _ = self.extract_var(
                    target_data, scaler_key, var_idx, self.sin_cos
                )
                target_ = target_.numpy()

            var_idx += index_shift

            if base_name == "ptrel":
                ht = new_data_dict[f"{collection}_ht"].reshape(-1, 1)
                fs_ = fs_ * ht
                target_ = target_ * ht if has_target else None
                base_name = "pt"
            if base_name == "vz_shifted" and unshift:
                fs_ = fs_ + pv_z
                target_ = target_ + pv_z if has_target else None
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
