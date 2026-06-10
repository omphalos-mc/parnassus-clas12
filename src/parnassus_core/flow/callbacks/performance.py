import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from parnassus_core.data.utils.scaling import Unscaler

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def normalize_phi(inp):
    return np.arctan2(np.sin(inp), np.cos(inp))


def _add_hist2d(fig, ax, pf_data, fs_data, bins, cmap):
    h, xe, ye, im = ax.hist2d(
        pf_data, fs_data, bins=bins, cmap=cmap, norm=mpl.colors.LogNorm()
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")


def _add_hist_1d(ax, data, bins, label, histtype="step", **kwargs):
    ax.hist(
        data,
        bins=bins,
        histtype=histtype,
        label=f"{label} {np.mean(data):.2f}/{np.std(data):.2f}",
        **kwargs,
    )


LOG_VARS = {
    "pt",
    "vx",
    "vy",
    "vz",
    "d0",
    "d0Error",
    "d0Sig",
    "z0",
    "z0Error",
    "z0Sig",
}

PARTICLE_BINS_DICT = {
    # Particle variables
    "pt": np.linspace(0, 100, 100),
    "eta": np.linspace(-3, 3, 100),
    "phi": np.linspace(-np.pi, np.pi, 100),
    "vx": np.linspace(-2, 2, 50),
    "vy": np.linspace(-2, 2, 50),
    "vz": np.linspace(-20, 20, 50),
}

EVENT_BINS_DICT = {
    "ht": np.linspace(0, 3000, 100),
    "met_x": np.linspace(-100, 100, 100),
    "met_y": np.linspace(-100, 100, 100),
    "npart": np.linspace(0.5, 400.5, 100),
}

IMPACT_BINS_DICT = {
    "d0": np.linspace(-0.5, 0.5, 50),
    "d0Error": np.linspace(0, 1, 50),
    "d0Sig": np.linspace(-2, 2, 50),
    "z0": np.linspace(-0.5, 0.5, 50),
    "z0Error": np.linspace(0, 2, 50),
    "z0Sig": np.linspace(-2, 2, 50),
}

BINS_DICT = {
    # Particle variables
    **PARTICLE_BINS_DICT,
    # Event variables
    **EVENT_BINS_DICT,
    # Impact parameters
    **IMPACT_BINS_DICT,
}

VARIABLES_LABELS = {
    "ht": "$H_T$",
    "met_x": "$MET_x$",
    "met_y": "$MET_y$",
    "npart": "$N_{part}$",
    "pt": "$p_T$",
    "eta": "$\\eta$",
    "phi": "$\\phi$",
    "vx": "$v_x$",
    "vy": "$v_y$",
    "vz": "$v_z$",
    "d0": "$d_0$",
    "z0": "$z_0$",
    "d0Error": "$\sigma_{d_0}$",
    "z0Error": "$\sigma_{z_0}$",
    "d0Sig": "$d_0 / \sigma_{d_0}$",
    "z0Sig": "$z_0 / \sigma_{z_0}$",
}


def subplots_with_absolute_sized_axes(
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    axis_width: float,
    axis_height: float,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Create axes with exact sizes.

    Spaces axes as far from each other and the figure edges as possible
    within the grid defined by nrows, ncols, and figsize.

    Allows you to share y and x axes, if desired.
    """
    fig = plt.figure(figsize=figsize, **kwargs)
    figwidth, figheight = figsize
    # spacing on each left and right side of the figure
    h_margin = (figwidth - (ncols * axis_width)) / figwidth / ncols / 2
    # spacing on each top and bottom of the figure
    v_margin = (figheight - (nrows * axis_height)) / figheight / nrows / 2
    row_addend = 1 / nrows
    col_addend = 1 / ncols
    inner_ax_width = axis_width / figwidth
    inner_ax_height = axis_height / figheight
    axes = []
    sharex_ax = None
    sharey_ax = None
    for row in range(nrows):
        bottom = (row * row_addend) + v_margin
        for col in range(ncols):
            left = (col * col_addend) + h_margin
            if not axes:
                axes.append(
                    fig.add_axes([left, bottom, inner_ax_width, inner_ax_height])
                )
                if sharex:
                    sharex_ax = axes[0]
                if sharey:
                    sharey_ax = axes[0]
            else:
                axes.append(
                    fig.add_axes(
                        [left, bottom, inner_ax_width, inner_ax_height],
                        sharex=sharex_ax,
                        sharey=sharey_ax,
                    )
                )
    return fig, np.flip(np.asarray(list(axes)).reshape((nrows, ncols)), axis=0)


class PerformanceLogger(Callback):
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []
        self.is_setted_up = False

        self.target_collection: str = ""
        self.ctxt_collection: str = ""

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        target_data, ctxt_dict = batch
        return_dict = {}
        with torch.no_grad():
            fs_data = pl_module.sample(
                target_data.shape,
                ctxt_dict=ctxt_dict,
            )
        for key, val in ctxt_dict.items():
            return_dict[key] = val.cpu()
        return_dict["fs_data"] = fs_data.cpu()
        return_dict["target_data"] = target_data.cpu()

        self.validation_step_outputs.append(return_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        ctxt_dict = {}
        fs_data = []
        target_data = []
        batch_size = self.validation_step_outputs[0]["fs_data"].shape[0]
        for el in self.validation_step_outputs:
            for key, val in el.items():
                if key == "fs_data":
                    fs_data.append(val)
                elif key == "target_data":
                    target_data.append(val)
                else:
                    if key not in ctxt_dict:
                        ctxt_dict[key] = []
                    ctxt_dict[key].append(val)
        fs_data = torch.cat(fs_data, dim=0)
        target_data = torch.cat(target_data, dim=0)
        for key, val in ctxt_dict.items():
            ctxt_dict[key] = torch.cat(val, dim=0)
        plot_dict = self.unscaler.unscale_variables(
            fs_data, ctxt_dict, target_data=target_data, unshift=False
        )
        pred_loss_dict = self.plot_data(pl_module=pl_module, plot_dict=plot_dict)
        for key, val in pred_loss_dict.items():
            pl_module.log(
                f"val/pred_{key}",
                val,
                batch_size=batch_size,
                sync_dist=True,
            )
        self.validation_step_outputs.clear()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module):
        if not self.is_setted_up:
            valid_dataset = trainer.val_dataloaders.dataset
            self.unscaler = Unscaler(
                feature_scaler=valid_dataset.feature_scaler,
                ctxt_global_vars=valid_dataset.ctxt_global_vars,
                ctxt_vars=valid_dataset.ctxt_vars,
                fs_vars=valid_dataset.fs_vars,
                sin_cos=valid_dataset.sin_cos,
            )
            self.target_collection = self.unscaler.target_collection
            self.ctxt_collection = self.unscaler.ctxt_collection
            self.is_setted_up = True
        self.validation_step_outputs.clear()

    def update_impact_vars(
        self, plot_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        assert self.target_collection == "pflow", (
            "Impact variables are only supported for pflow collection."
        )
        for var in IMPACT_BINS_DICT.keys():
            if f"pflow_{var}" not in plot_dict:
                if var in {"d0", "z0"}:
                    plot_dict[f"pflow_{var}"] = (
                        plot_dict[f"pflow_{var}Sig"] * plot_dict[f"pflow_{var}Error"]
                    )
                    plot_dict[f"fs_{var}"] = (
                        plot_dict[f"fs_{var}Sig"] * plot_dict[f"fs_{var}Error"]
                    )
                elif var in {"d0Error", "z0Error"}:
                    var_ = var.replace("Error", "")
                    plot_dict[f"pflow_{var}"] = (
                        plot_dict[f"pflow_{var_}"] / plot_dict[f"pflow_{var_}Sig"]
                    )
                    plot_dict[f"fs_{var}"] = (
                        plot_dict[f"fs_{var_}"] / plot_dict[f"fs_{var_}Sig"]
                    )
                elif var in {"d0Sig", "z0Sig"}:
                    var_ = var.replace("Sig", "")
                    plot_dict[f"pflow_{var}"] = (
                        plot_dict[f"pflow_{var_}"] / plot_dict[f"pflow_{var_}Error"]
                    )
                    plot_dict[f"fs_{var}"] = (
                        plot_dict[f"fs_{var_}"] / plot_dict[f"fs_{var_}Error"]
                    )
        return plot_dict

    def _save_and_log_figure(self, fig, plot_name: str, pl_module: LightningModule):
        """Save figure to file and log to experiment tracker."""
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.asarray(canvas.buffer_rgba())[..., :3]
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.savefig(f"logs/figs/{plot_name}_{pl_module.current_epoch}.png")
        if pl_module.logger is not None:
            pl_module.logger.experiment.log_image(
                image_data=image,
                name=f"ED_{pl_module.current_epoch}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"{plot_name}_{pl_module.current_epoch}.png")
        plt.close(fig)

    def _add_comparison_histogram(
        self,
        ax,
        target_data: np.ndarray,
        fs_data: np.ndarray,
        bins: np.ndarray,
        target_label: str,
        fs_label: str = "Fastsim",
        log_scale: bool = False,
    ):
        """Add overlaid histograms comparing target and fastsim data."""
        _add_hist_1d(
            ax, target_data, bins, target_label, histtype="stepfilled", alpha=0.5
        )
        _add_hist_1d(ax, fs_data, bins, fs_label, histtype="step")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()

    def plot_data(
        self, pl_module: LightningModule, plot_dict: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Plot performance data and return metrics dict."""
        return {}


class ParticlePerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        plot_dict, bins_dict, plot_name = self._prepare_plot_data(plot_dict)
        fig, axs = self._create_figure(bins_dict)
        ax_iter = iter(axs.flatten())

        fs_mask = plot_dict["fs_mask"].astype(bool)

        self._plot_particle_variables(plot_dict, bins_dict, ax_iter, fs_mask)
        self._plot_class_or_charge(plot_dict, next(ax_iter), fs_mask)
        event_metrics = self._plot_event_metrics(plot_dict, ax_iter, fs_mask)

        for ax in axs.flatten():
            ax.set_box_aspect(1)
        self._save_and_log_figure(fig, plot_name, pl_module)
        return event_metrics

    def _prepare_plot_data(self, plot_dict):
        """Prepare data and determine bins/plot configuration."""
        bins_dict = PARTICLE_BINS_DICT.copy()
        plot_name = "PART_ED"

        target_collection = self.target_collection

        if "pflow_d0" in plot_dict or "pflow_d0Error" in plot_dict:
            plot_dict = self.update_impact_vars(plot_dict)
            bins_dict.update(IMPACT_BINS_DICT)
            plot_name = "PART_IMP_ED"

        if f"{target_collection}_vz_shifted" in plot_dict:
            bins_dict["vz"] = np.linspace(-0.1, 0.1, 200)
            plot_dict[f"{target_collection}_vz"] = plot_dict.pop(
                f"{target_collection}_vz_shifted"
            )
            plot_dict["fs_vz"] = plot_dict.pop("fs_vz_shifted")

        return plot_dict, bins_dict, plot_name

    def _create_figure(self, bins_dict):
        """Create figure with appropriate number of subplots."""
        # Particle vars + 1 class/charge + 5 event metrics (HT, HT_res, MET_x, MET_y, Class)
        num_plots = len(bins_dict) + 6
        num_cols = 6
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axs = subplots_with_absolute_sized_axes(
            num_rows,
            num_cols,
            figsize=(num_cols * 5, num_rows * 5),
            axis_width=4,
            axis_height=4,
            dpi=200,
        )
        return fig, axs

    def _plot_particle_variables(self, plot_dict, bins_dict, ax_iter, fs_mask):
        """Plot each particle variable with proper filtering."""
        target_collection = self.target_collection
        charged_only_vars = {
            "vx",
            "vy",
            "vz",
            "d0",
            "d0Error",
            "d0Sig",
            "z0",
            "z0Error",
            "z0Sig",
        }

        for var, bins in bins_dict.items():
            ax = next(ax_iter)
            ax.set_title(VARIABLES_LABELS[var])

            target_data = plot_dict[f"{target_collection}_{var}"][fs_mask]
            fs_data = plot_dict[f"fs_{var}"][fs_mask]

            if var in charged_only_vars:
                charge_mask = self._get_charged_particle_mask(
                    plot_dict, target_collection, fs_mask
                )
                target_data = target_data[charge_mask]
                fs_data = fs_data[charge_mask]

            self._add_comparison_histogram(
                ax,
                target_data,
                fs_data,
                bins,
                target_collection.capitalize(),
                log_scale=(var in LOG_VARS),
            )

    def _get_charged_particle_mask(self, plot_dict, target_collection, fs_mask):
        """Get mask for charged particles only (class < 3)."""
        target_charged = plot_dict[f"{target_collection}_class"][fs_mask] < 3
        fs_charged = plot_dict["fs_class"][fs_mask] < 3
        return target_charged & fs_charged

    def _plot_class_or_charge(self, plot_dict, ax, fs_mask):
        """Plot either charge histogram or class confusion matrix."""
        target_collection = self.target_collection
        ctxt_collection = self.ctxt_collection

        if "fs_charge" in plot_dict:
            class_bins = np.arange(-1, 3) - 0.5
            ax.hist(
                plot_dict[f"{target_collection}_charge"][fs_mask],
                bins=class_bins,
                histtype="stepfilled",
                label=target_collection.capitalize(),
                density=True,
                alpha=0.5,
            )
            ax.hist(
                plot_dict["fs_charge"][fs_mask],
                bins=class_bins,
                histtype="step",
                label="Fastsim",
                density=True,
            )
            ax.set_xticks([-1, 0, 1], [-1, 0, 1])
            ax.set_title("Charge")
            ax.legend()
        else:
            cm = confusion_matrix(
                plot_dict[f"{target_collection}_class"][fs_mask],
                plot_dict["fs_class"][fs_mask],
                normalize="true",
            )
            df_cm = pd.DataFrame(
                cm,
                index=["Ch had", "El", "Mu", "Neut had", "Phot"][: cm.shape[0]],
                columns=["Ch had", "El", "Mu", "Neut had", "Phot"][: cm.shape[1]],
            )
            sns.heatmap(
                df_cm,
                annot=True,
                annot_kws={"size": 8},
                cmap=sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
                cbar=False,
                ax=ax,
            )
            ax.set_title("classes")
            ax.set_ylabel(ctxt_collection.capitalize())
            ax.set_xlabel(target_collection.capitalize())

    def _plot_event_metrics(self, plot_dict, ax_iter, fs_mask):
        """Plot HT, MET, and class distributions. Returns computed metrics."""
        target_collection = self.target_collection
        ctxt_collection = self.ctxt_collection
        ctxt_mask = plot_dict["ctxt_mask"]

        # Compute event quantities
        ctxt_pt = plot_dict[f"{ctxt_collection}_pt"] * ctxt_mask
        ctxt_phi = plot_dict[f"{ctxt_collection}_phi"] * ctxt_mask
        ctxt_ht = ctxt_pt.sum(-1)
        ctxt_met_x = (ctxt_pt * np.cos(ctxt_phi)).sum(-1)
        ctxt_met_y = (ctxt_pt * np.sin(ctxt_phi)).sum(-1)

        target_pt = plot_dict[f"{target_collection}_pt"] * fs_mask
        target_phi = plot_dict[f"{target_collection}_phi"] * fs_mask
        target_ht = target_pt.sum(-1)
        target_met_x = (target_pt * np.cos(target_phi)).sum(-1)
        target_met_y = (target_pt * np.sin(target_phi)).sum(-1)

        fs_pt = plot_dict["fs_pt"] * fs_mask
        fs_phi = plot_dict["fs_phi"] * fs_mask
        fs_ht = fs_pt.sum(-1)
        fs_met_x = (fs_pt * np.cos(fs_phi)).sum(-1)
        fs_met_y = (fs_pt * np.sin(fs_phi)).sum(-1)

        # HT plot
        ht_bins = np.linspace(0, 3000, 100)
        ax = next(ax_iter)
        _add_hist_1d(
            ax,
            target_ht,
            ht_bins,
            target_collection.capitalize(),
            histtype="stepfilled",
            alpha=0.5,
        )
        _add_hist_1d(ax, ctxt_ht, ht_bins, ctxt_collection.capitalize())
        _add_hist_1d(ax, fs_ht, ht_bins, "Fastsim")
        ax.set_title("Jet pT")
        ax.legend()

        # HT residual
        ax = next(ax_iter)
        self._add_comparison_histogram(
            ax,
            target_ht - ctxt_ht,
            fs_ht - ctxt_ht,
            np.linspace(-200, 200, 100),
            target_collection.capitalize(),
        )
        ax.set_title("Jet pT Residual")

        # MET x
        met_bins = np.linspace(-100, 100, 100)
        ax = next(ax_iter)
        _add_hist_1d(
            ax,
            target_met_x,
            met_bins,
            target_collection.capitalize(),
            histtype="stepfilled",
            alpha=0.5,
        )
        _add_hist_1d(ax, ctxt_met_x, met_bins, ctxt_collection.capitalize())
        _add_hist_1d(ax, fs_met_x, met_bins, "Fastsim")
        ax.set_title("MET x")
        ax.legend()

        # MET y
        ax = next(ax_iter)
        _add_hist_1d(
            ax,
            target_met_y,
            met_bins,
            target_collection.capitalize(),
            histtype="stepfilled",
            alpha=0.5,
        )
        _add_hist_1d(ax, ctxt_met_y, met_bins, ctxt_collection.capitalize())
        _add_hist_1d(ax, fs_met_y, met_bins, "Fastsim")
        ax.set_title("MET y")
        ax.legend()

        # Class distribution
        ax = next(ax_iter)
        class_bins = np.linspace(-0.5, 4.5, 6)
        ax.hist(
            plot_dict[f"{target_collection}_class"][fs_mask],
            bins=class_bins,
            histtype="stepfilled",
            label=target_collection.capitalize(),
            density=True,
            alpha=0.5,
        )
        ax.hist(
            plot_dict["fs_class"][fs_mask],
            bins=class_bins,
            histtype="step",
            label="Fastsim",
            density=True,
        )
        ax.set_xticks(np.arange(5), ["Ch had", "El", "Mu", "Neut had", "Phot"])
        ax.set_title("Class")
        ax.legend()

        return {
            "ht": np.mean((fs_ht - target_ht) ** 2),
            "met_x": np.mean((fs_met_x - target_met_x) ** 2),
            "met_y": np.mean((fs_met_y - target_met_y) ** 2),
        }


class EventPerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        fig, axs = subplots_with_absolute_sized_axes(
            4, 2, figsize=(18, 18), axis_width=8, axis_height=4, dpi=200
        )
        target_collection = self.target_collection
        ctxt_collection = self.ctxt_collection

        res_bins = {
            "ht": np.linspace(-2, 1, 50),
            "npart": np.linspace(-250, 150, 50),
            "met_x": np.linspace(-100, 100, 50),
            "met_y": np.linspace(-100, 100, 50),
        }
        for i, var in enumerate(["ht", "met_x", "met_y", "npart"]):
            _add_hist_1d(
                axs[i][0],
                plot_dict[f"{target_collection}_{var}"],
                bins=EVENT_BINS_DICT[var],
                label=target_collection.capitalize(),
                histtype="stepfilled",
                alpha=0.5,
            )
            _add_hist_1d(
                axs[i][0],
                plot_dict[f"fs_{var}"],
                bins=EVENT_BINS_DICT[var],
                label="Fastsim",
                histtype="step",
            )
            _add_hist_1d(
                axs[i][0],
                plot_dict[f"{ctxt_collection}_{var}"],
                bins=EVENT_BINS_DICT[var],
                label=ctxt_collection.capitalize(),
                histtype="step",
                color="k",
            )
            axs[i][0].set_title(VARIABLES_LABELS[var])
            axs[i][0].legend()

            target_res = (
                plot_dict[f"{target_collection}_{var}"]
                - plot_dict[f"{ctxt_collection}_{var}"]
            )
            fs_res = plot_dict[f"fs_{var}"] - plot_dict[f"{ctxt_collection}_{var}"]
            if var == "ht":
                target_res /= plot_dict[f"{ctxt_collection}_ht"]
                fs_res /= plot_dict[f"{ctxt_collection}_ht"]

            self._add_comparison_histogram(
                axs[i][1],
                target_res,
                fs_res,
                res_bins[var],
                f"{target_collection.capitalize()} - {ctxt_collection.capitalize()}",
                f"Fastsim - {ctxt_collection.capitalize()}",
            )
            axs[i][1].set_title(f"{VARIABLES_LABELS[var]} Residuals")

        self._save_and_log_figure(fig, "EVT_ED", pl_module)
        return {
            var: np.mean(
                (plot_dict[f"fs_{var}"] - plot_dict[f"{target_collection}_{var}"]) ** 2
            )
            for var in ["ht", "met_x", "met_y", "npart"]
        }


class ImpactPerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        fig, axs = subplots_with_absolute_sized_axes(
            2, 3, figsize=(20, 10), axis_width=6, axis_height=4, dpi=200
        )
        axs = axs.flatten()
        target_collection = self.target_collection

        pf_mask = plot_dict["fs_mask"].astype(bool)
        for key, val in plot_dict.items():
            if (
                (key.startswith(f"{target_collection}_") or key.startswith("fs_"))
                and key != "fs_mask"
                and val.ndim == 2
            ):
                plot_dict[key] = val[pf_mask]
        plot_dict = self.update_impact_vars(plot_dict)

        for i, (var, bins) in enumerate(IMPACT_BINS_DICT.items()):
            self._add_comparison_histogram(
                axs[i],
                plot_dict[f"{target_collection}_{var}"],
                plot_dict[f"fs_{var}"],
                bins,
                target_collection.capitalize(),
                "Fastsim",
                log_scale=(var in LOG_VARS),
            )
            axs[i].set_xlabel(VARIABLES_LABELS[var])
            axs[i].set_ylabel("Counts")
            axs[i].set_title(var)

        self._save_and_log_figure(fig, "IMP_ED", pl_module)
        return {
            var: np.mean(
                (plot_dict[f"fs_{var}"] - plot_dict[f"{target_collection}_{var}"]) ** 2
            )
            for var in ["d0", "d0Error", "d0Sig", "z0", "z0Error", "z0Sig"]
            if f"fs_{var}" in plot_dict
        }
