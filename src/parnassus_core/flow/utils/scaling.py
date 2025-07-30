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
