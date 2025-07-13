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
        self.power = self.config.get("power", 1.0)

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
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        match self.fn:
            case "log" | "log1p" | "sqrt" | "tanh" | "asinh" | "atan":
                x = getattr(torch, self.fn)(x / self.fn_scale)
            case "pow":
                x = torch.sign(x) * torch.pow(torch.abs(x) / self.fn_scale, self.power)
            case _:
                x = x / self.fn_scale

        return (x - shift) / scale

    def inverse_transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
        match self.fn:
            case "log":
                return torch.exp(x) * self.fn_scale
            case "log1p":
                return torch.expm1(x) * self.fn_scale
            case "sqrt":
                return torch.pow(x, 2) * self.fn_scale
            case "tanh":
                return torch.atanh(x) * self.fn_scale
            case "asinh":
                return torch.sinh(x) * self.fn_scale
            case "atan":
                return torch.tan(x) * self.fn_scale
            case "pow":
                return torch.sign(x) * torch.pow(torch.abs(x), 1 / self.power)
            case _:
                # For other functions, we assume they are linear transformations
                # and do not need to be inverted.
                return x * self.fn_scale
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
