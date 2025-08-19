from collections import defaultdict
from typing import ClassVar, Generic, Literal, Type
import numpy as np
import matplotlib.pyplot as plt


from .labels import VAR_LABELS, RESIDUAL_VAR_LABELS, SAMPLE_LABELS
from .bins import BinsConfig
from .config import SampleT, Source
from .performance import Performance

EVENT_VARS = ["npart", "njet", "met_x", "met_y", "ht"]

JET_VARS = ["pt", "eta", "phi", "d2", "c2"]

PARTICLE_VARS = [
    "pt",
    "eta",
    "phi",
    "vx",
    "vy",
    "vz",
]

IMPACT_VARS = [
    "d0",
    "d0Error",
    "d0Sig",
    "z0",
    "z0Error",
    "z0Sig",
    "ip3d",
    "ip3dError",
    "ip3dSig",
]

FONT_SIZE = 15
AX_HEIGHT = 5
AX_WIDTH = 5

DEFAULT_STYLE_DICT = defaultdict(lambda: dict())


class PlotHelper(Generic[SampleT]):
    bins_config: ClassVar[Type[BinsConfig[SampleT]]]

    def __init__(
        self,
        performance: Performance,
        plot_truth: bool = False,
        font_size: int = FONT_SIZE,
        ax_height: int = AX_HEIGHT,
        ax_width: int = AX_WIDTH,
        style_dict: dict | None = None,
    ):
        self.performance = performance
        self.plot_truth = plot_truth
        self.num_samples = len(performance.sample_data_dict)

        self.font_size = font_size
        self.ax_height = ax_height
        self.ax_width = ax_width
        self.style_dict = style_dict if style_dict is not None else DEFAULT_STYLE_DICT

    def plot_features(
        self,
        data_dict_label: str,
        var_list: list[str],
        bins: dict[SampleT, dict[str, np.ndarray]] | None = None,
        log_scale_vars: list[str] | str = [],
        xlabel_prefix: str = "",
        particle_mask_type: str = "all",
    ):
        var_labels = VAR_LABELS | RESIDUAL_VAR_LABELS
        fig, ax = plt.subplots(
            self.num_samples,
            len(var_list),
            figsize=(len(var_list) * self.ax_width, self.num_samples * self.ax_height),
        )
        if len(ax.shape) == 1:
            ax = ax.reshape((self.num_samples, len(var_list)))
        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            num_legends = len(sample_data.data_dict) - 1 * (not self.plot_truth)
            for j, var in enumerate(var_list):
                for pf_data in sample_data.data_dict.values():
                    if not self.plot_truth and (pf_data.source == Source.TR):
                        continue
                    data_dict = getattr(pf_data, data_dict_label)
                    value = data_dict[var]
                    if value.dtype == np.object_:
                        value = np.concatenate(value)
                    if var == "d2" and not var.endswith("_res"):
                        value = np.log(value + 1e-9)
                    if "particle" in data_dict_label:
                        match particle_mask_type:
                            case "all":
                                mask = np.ones_like(value, dtype=bool)
                            case "had":
                                mask = np.concatenate(data_dict["class"]) == 0
                            case "el":
                                mask = np.concatenate(data_dict["class"]) == 1
                            case "mu":
                                mask = np.concatenate(data_dict["class"]) == 2
                            case "charged":
                                mask = np.concatenate(data_dict["class"]) < 3
                            case "neutral":
                                mask = np.concatenate(data_dict["class"]) >= 3
                            case _:
                                raise ValueError(
                                    f"Unknown particle_mask_type: {particle_mask_type}"
                                )
                        value = value[mask]
                    if var in IMPACT_VARS:
                        value = value[value > -990]
                    ax[i, j].hist(
                        value,
                        bins=bins[var][sample],
                        density=True,
                        histtype="stepfilled"
                        if pf_data.source == Source.PF
                        else "step",
                        label=pf_data.label,
                        **self.style_dict[pf_data.name],
                    )

                ax[i, j].yaxis.tick_right()
                if var in log_scale_vars:
                    ax[i, j].set_yscale("log")
                if i == self.num_samples - 1:
                    ax[i, j].set_xlabel(
                        xlabel_prefix + var_labels[var], fontsize=self.font_size
                    )
            ax[i, 0].annotate(
                SAMPLE_LABELS[sample.name.lower()],
                xy=(-0.05, 0.5),
                xycoords="axes fraction",
                fontsize=self.font_size,
                ha="center",
                va="center",
                rotation=90,
            )
            ax[i, -1].set_ylabel("a.u.", fontsize=self.font_size)
            ax[i, -1].yaxis.set_label_position("right")

        ax[0, 2].legend(
            loc="upper center",
            ncol=num_legends,
            fancybox=True,
            fontsize=self.font_size,
            bbox_to_anchor=(0.5, 1.325),
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.15)
        return fig

    def plot_event_features(
        self,
        bins: dict[SampleT, dict[str, np.ndarray]] | None = None,
        residuals: bool = False,
    ):
        if residuals:
            bins = self.bins_config.event_res_bins if bins is None else bins
            event_vars = [f"{var}_res" for var in EVENT_VARS]
        else:
            bins = self.bins_config.event_bins if bins is None else bins
            event_vars = EVENT_VARS
        fig = self.plot_features(
            "event_data",
            event_vars,
            bins=bins,
        )
        return fig

    def plot_jet_features(
        self,
        bins: dict[SampleT, dict[str, np.ndarray]] | None = None,
        residuals: bool = False,
        log_scale_vars: list[str] | str = [],
    ):
        if residuals:
            bins = self.bins_config.jet_res_bins if bins is None else bins
            jet_vars = [f"{var}_res" for var in JET_VARS]
        else:
            bins = self.bins_config.jet_bins if bins is None else bins
            jet_vars = JET_VARS
        if log_scale_vars == "all":
            log_scale_vars = jet_vars
        xlabel_prefix = "Residual jet " if residuals else "Jet "
        fig = self.plot_features(
            "jet_data",
            jet_vars,
            bins=bins,
            log_scale_vars=log_scale_vars,
            xlabel_prefix=xlabel_prefix,
        )
        return fig

    def plot_particle_features(
        self,
        bins: dict[SampleT, dict[str, np.ndarray]] | None = None,
        collection: str = "all",
        residuals: bool = False,
        log_scale_vars: list[str] | str = [],
        particle_mask_type: str = "all",
    ):
        if residuals:
            bins = self.bins_config.particle_res_bins if bins is None else bins
            particle_vars = [f"{var}_res" for var in PARTICLE_VARS]
        else:
            bins = self.bins_config.particle_bins if bins is None else bins
            particle_vars = PARTICLE_VARS
        if log_scale_vars == "all":
            log_scale_vars = particle_vars
        xlabel_prefix = "Residual particle " if residuals else "Particle "
        fig = self.plot_features(
            f"particle_data_{collection}",
            particle_vars,
            bins=bins,
            log_scale_vars=log_scale_vars,
            xlabel_prefix=xlabel_prefix,
            particle_mask_type=particle_mask_type,
        )
        return fig

    def plot_impact_features(
        self,
        bins: dict[SampleT, dict[str, np.ndarray]] | None = None,
        collection: str = "all",
        log_scale_vars: list[str] | str = [],
        particle_mask_type: Literal["had", "el", "mu", "charged"] = "charged",
    ):
        bins = self.bins_config.particle_bins if bins is None else bins
        impact_vars = IMPACT_VARS
        if log_scale_vars == "all":
            log_scale_vars = impact_vars
        fig = self.plot_features(
            f"particle_data_{collection}",
            impact_vars,
            bins=bins,
            xlabel_prefix="Impact parameter ",
            log_scale_vars=log_scale_vars,
            particle_mask_type=particle_mask_type,
        )
        return fig
