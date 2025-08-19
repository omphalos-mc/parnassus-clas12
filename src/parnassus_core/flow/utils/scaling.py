import torch
from collections import defaultdict

TRANSFORM_TYPES = {
    "std",
    "min_max",
    "min_max_sym",
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
        self.fn_scale = self.config.get("fn_scale", 1.0)
        self.fn_shift = self.config.get("fn_shift", 0.0)
        self.power = self.config.get("power", 1.0)
        self.clip = self.config.get("clip", None)

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

    def transform(self, x, shift=None, scale=None):
        if self.clip is not None:
            x = torch.clamp(x, min=self.min_clip, max=self.max_clip)
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = (x + self.fn_shift) / self.fn_scale
        match self.fn:
            case "log" | "log1p" | "sqrt" | "tanh" | "asinh" | "atan":
                x = getattr(torch, self.fn)(x)
            case "pow":
                x = torch.sign(x) * torch.pow(torch.abs(x), self.power)
            case _:
                pass
        return (x - shift) / scale

    def inverse_transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
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

    def transform(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply the scaling transformations to the input features.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of features to be transformed.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of transformed features.
        """
        for key, value in x.items():
            if key in self.transforms:
                x[key] = self.transforms[key].transform(value)
        return x

    def inverse_transform(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply the inverse scaling transformations to the input features.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of features to be inverse transformed.
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of inverse transformed features.
        """
        for key, value in x.items():
            if key in self.transforms:
                x[key] = self.transforms[key].inverse_transform(value)
        return x


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

    def extract_var(self, data, var_name, var_idx, sin_cos=False):
        index_shift = 0
        if var_name.endswith("phi") and sin_cos:
            var = torch.atan2(
                data[..., var_idx],
                data[..., var_idx + 1],
            )
            index_shift = 2
        elif var_name.endswith("class"):
            var = data[..., var_idx : var_idx + 5].argmax(-1)
            index_shift = 5
        else:
            var = self.feature_scaler[var_name].inverse_transform(data[..., var_idx])
            index_shift = 1
        return var, index_shift

    def unscale_variables(self, fs_data, ctxt_dict, pf_data=None):
        fs_data = fs_data.cpu()
        has_pflow = False
        if pf_data is not None:
            pf_data = pf_data.cpu()
            has_pflow = True
        ctxt_dict = {k: v.cpu() for k, v in ctxt_dict.items()}
        new_data_dict = {}
        var_idx = 0
        for var_name in self.ctxt_global_vars:
            if var_name == "means":
                var_idx += len(
                    [
                        el
                        for el in self.ctxt_vars
                        if (el.startswith("truth_") and "class" not in el)
                    ]
                )
            elif var_name.startswith("truth_") or var_name.startswith("ntruth"):
                var_name = var_name.replace("truth_", "").replace("ntruth", "npart")
                new_data_dict[f"tr_{var_name}"] = (
                    self.feature_scaler[var_name]
                    .inverse_transform(ctxt_dict["ctxt_global_data"][..., var_idx])
                    .numpy()
                )
                var_idx += 1
            elif var_name.startswith("pflow_"):
                var_name = var_name.replace("pflow_", "")
                new_data_dict[f"pf_{var_name}"] = (
                    self.feature_scaler[var_name]
                    .inverse_transform(ctxt_dict["ctxt_global_data"][..., var_idx])
                    .numpy()
                )
                var_idx += 1
        var_idx = 0
        for var_name in self.ctxt_vars:
            if var_name.startswith("pflow_"):
                continue
            var_name = var_name.replace("truth_", "")
            var, index_shift = self.extract_var(
                ctxt_dict["ctxt_data"], var_name, var_idx, self.sin_cos
            )
            var_idx += index_shift
            var = var.numpy()
            if var_name == "ptrel":
                var = var * new_data_dict["tr_ht"].reshape(-1, 1)
                var_name = "pt"
            new_data_dict[f"tr_{var_name}"] = var
        var_idx = 0
        for var_name in self.fs_vars:
            var_name = var_name.replace("pflow_", "").replace("npflow", "npart")
            fs_, index_shift = self.extract_var(
                fs_data, var_name, var_idx, self.sin_cos
            )
            if has_pflow:
                pf_, _ = self.extract_var(pf_data, var_name, var_idx, self.sin_cos)
            var_idx += index_shift
            fs_ = fs_.numpy()
            pf_ = pf_.numpy() if has_pflow else None
            if var_name == "ptrel":
                fs_ = fs_ * new_data_dict["pf_ht"].reshape(-1, 1)
                pf_ = pf_ * new_data_dict["pf_ht"].reshape(-1, 1) if has_pflow else None
                var_name = "pt"
            new_data_dict[f"fs_{var_name}"] = fs_
            if has_pflow:
                new_data_dict[f"pf_{var_name}"] = pf_
        if "ctxt_mask" in ctxt_dict:
            new_data_dict["ctxt_mask"] = ctxt_dict["ctxt_mask"].numpy()
        if "fs_mask" in ctxt_dict:
            new_data_dict["fs_mask"] = ctxt_dict["fs_mask"].numpy()
        if "eventNumber" in ctxt_dict:
            new_data_dict["eventNumber"] = ctxt_dict["eventNumber"].numpy()
        return new_data_dict
