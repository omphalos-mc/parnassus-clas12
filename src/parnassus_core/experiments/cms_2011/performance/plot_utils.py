from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


from .labels import VAR_LABLES, RESIDUAL_VAR_LABELS, SAMPLE_LABELS
from .bins import (
    DEFAULT_EVENT_BINS,
    DEFAULT_EVENT_RES_BINS,
    DEFAULT_JET_BINS,
    DEFAULT_JET_RES_BINS,
    DEFAULT_PARTICLE_BINS,
    DEFAULT_PARTICLE_RES_BINS,
)
from .config import Sample, Source
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
    "z0",
    "z0Error",
]

AX_HEIGHT = 5
AX_WIDTH = 5

DEFAULT_STYLE_DICT = defaultdict(lambda: dict())


class PlotHelper:
    def __init__(
        self,
        performance: Performance,
        plot_truth: bool = False,
        style_dict: dict | None = None,
    ):
        self.performance = performance
        self.plot_truth = plot_truth
        self.num_samples = len(performance.sample_data_dict)
        self.style_dict = style_dict if style_dict is not None else DEFAULT_STYLE_DICT

    def plot_event_features(
        self,
        bins: dict[Sample, dict[str, np.ndarray]] | None = None,
        residuals: bool = False,
    ):
        if residuals:
            bins = DEFAULT_EVENT_RES_BINS if bins is None else bins
            event_vars = [f"{var}_res" for var in EVENT_VARS]
            var_labels = RESIDUAL_VAR_LABELS
        else:
            bins = DEFAULT_EVENT_BINS if bins is None else bins
            event_vars = EVENT_VARS
            var_labels = VAR_LABLES
        fig, ax = plt.subplots(
            self.num_samples,
            len(event_vars),
            figsize=(len(event_vars) * AX_HEIGHT, self.num_samples * AX_WIDTH),
        )
        if len(ax.shape) == 1:
            ax = ax.reshape((self.num_samples, len(event_vars)))
        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            num_legends = len(sample_data.data_dict) - 1 * (not self.plot_truth)
            for j, var in enumerate(event_vars):
                for pf_data in sample_data.data_dict.values():
                    if not self.plot_truth and (pf_data.source == Source.TR):
                        continue
                    value = pf_data.event_data[var]
                    label = pf_data.label
                    ax[i, j].hist(
                        value,
                        bins=bins[var][sample],
                        density=True,
                        histtype="stepfilled"
                        if pf_data.source == Source.PF
                        else "step",
                        label=label,
                        **self.style_dict[pf_data.name],
                    )

                ax[i, j].yaxis.tick_right()
                if i == self.num_samples - 1:
                    ax[i, j].set_xlabel(var_labels[var])
            ax[i, 0].annotate(
                SAMPLE_LABELS[sample.name.lower()],
                xy=(-0.05, 0.5),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
                va="center",
                rotation=90,
            )
            ax[i, -1].set_ylabel("a.u.", fontsize=12)
            ax[i, -1].yaxis.set_label_position("right")

        ax[0, 2].legend(
            loc="upper center",
            ncol=num_legends,
            fancybox=True,
            fontsize=12,
            bbox_to_anchor=(0.5, 1.325),
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.15)
        return fig

    def plot_jet_features(
        self,
        bins: dict[Sample, dict[str, np.ndarray]] | None = None,
        residuals: bool = False,
        log_scale_vars: list[str] | str = [],
    ):
        if residuals:
            bins = DEFAULT_JET_RES_BINS if bins is None else bins
            jet_vars = [f"{var}_res" for var in JET_VARS]
            var_labels = RESIDUAL_VAR_LABELS
        else:
            bins = DEFAULT_JET_BINS if bins is None else bins
            jet_vars = JET_VARS
            var_labels = VAR_LABLES
        if log_scale_vars == "all":
            log_scale_vars = jet_vars
        fig, ax = plt.subplots(
            self.num_samples,
            len(jet_vars),
            figsize=(len(jet_vars) * AX_HEIGHT, self.num_samples * AX_WIDTH),
        )
        if len(ax.shape) == 1:
            ax = ax.reshape((self.num_samples, len(jet_vars)))
        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            num_legends = len(sample_data.data_dict) - 1 * (not self.plot_truth)
            for j, var in enumerate(jet_vars):
                for pf_data in sample_data.data_dict.values():
                    if not self.plot_truth and (pf_data.source == Source.TR):
                        continue
                    value = pf_data.jet_data[var]
                    if value.dtype == np.object_:
                        value = np.concatenate(value)
                    if var == "d2" and not residuals:
                        value = np.log(value + 1e-9)
                    label = pf_data.label
                    ax[i, j].hist(
                        value,
                        bins=bins[var][sample],
                        density=True,
                        histtype="stepfilled"
                        if pf_data.source == Source.PF
                        else "step",
                        label=label,
                        **self.style_dict[pf_data.name],
                    )

                ax[i, j].yaxis.tick_right()
                if i == self.num_samples - 1:
                    if residuals:
                        ax[i, j].set_xlabel(f"Residual jet {var_labels[var]}")
                    else:
                        ax[i, j].set_xlabel(f"Jet {var_labels[var]}")
                if var in log_scale_vars:
                    ax[i, j].set_yscale("log")
            ax[i, 0].annotate(
                SAMPLE_LABELS[sample.name.lower()],
                xy=(-0.05, 0.5),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
                va="center",
                rotation=90,
            )
            ax[i, -1].set_ylabel("a.u.", fontsize=12)
            ax[i, -1].yaxis.set_label_position("right")

        ax[0, 2].legend(
            loc="upper center",
            ncol=num_legends,
            fancybox=True,
            fontsize=12,
            bbox_to_anchor=(0.5, 1.325),
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.15)
        return fig

    def plot_particle_features(
        self,
        bins: dict[Sample, dict[str, np.ndarray]] | None = None,
        collection: str = "all",
        residuals: bool = False,
        log_scale_vars: list[str] | str = [],
    ):
        if residuals:
            bins = DEFAULT_PARTICLE_RES_BINS if bins is None else bins
            particle_vars = [f"{var}_res" for var in PARTICLE_VARS]
            var_labels = RESIDUAL_VAR_LABELS
        else:
            bins = DEFAULT_PARTICLE_BINS if bins is None else bins
            particle_vars = PARTICLE_VARS
            var_labels = VAR_LABLES
        if log_scale_vars == "all":
            log_scale_vars = particle_vars
        fig, ax = plt.subplots(
            self.num_samples,
            len(particle_vars),
            figsize=(len(particle_vars) * AX_HEIGHT, self.num_samples * AX_WIDTH),
        )
        if len(ax.shape) == 1:
            ax = ax.reshape((self.num_samples, len(particle_vars)))
        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            num_legends = len(sample_data.data_dict) - 1 * (not self.plot_truth)
            for j, var in enumerate(particle_vars):
                for pf_data in sample_data.data_dict.values():
                    if not self.plot_truth and (pf_data.source == Source.TR):
                        continue
                    # value = pf_data.particle_data[var]
                    value = getattr(pf_data, f"particle_data_{collection}")[var]
                    if value.dtype == np.object_:
                        value = np.concatenate(value)
                    if var == "d2" and not residuals:
                        value = np.log(value + 1e-9)
                    label = pf_data.label
                    ax[i, j].hist(
                        value,
                        bins=bins[var][sample],
                        density=True,
                        histtype="stepfilled"
                        if pf_data.source == Source.PF
                        else "step",
                        label=label,
                        **self.style_dict[pf_data.name],
                    )

                ax[i, j].yaxis.tick_right()
                if i == self.num_samples - 1:
                    if residuals:
                        ax[i, j].set_xlabel(f"Residual particle {var_labels[var]}")
                    else:
                        ax[i, j].set_xlabel(f"Particle {var_labels[var]}")
                if var in log_scale_vars:
                    ax[i, j].set_yscale("log")
            ax[i, 0].annotate(
                SAMPLE_LABELS[sample.name.lower()],
                xy=(-0.05, 0.5),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
                va="center",
                rotation=90,
            )
            ax[i, -1].set_ylabel("a.u.", fontsize=12)
            ax[i, -1].yaxis.set_label_position("right")

        ax[0, 2].legend(
            loc="upper center",
            ncol=num_legends,
            fancybox=True,
            fontsize=12,
            bbox_to_anchor=(0.5, 1.325),
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.15)
        return fig
