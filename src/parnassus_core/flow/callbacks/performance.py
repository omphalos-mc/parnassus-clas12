from pathlib import Path

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
from parnassus_core.utils.pid import (
    pdgid_to_class,
    pdgid_to_onehot_index,
    pid_onehot_ids,
)

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
    # A diverged sample (inf) makes auto-binning raise on that rank alone, which
    # leaves every other rank waiting in the next collective until NCCL aborts.
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    ax.hist(
        data,
        bins=bins,
        histtype=histtype,
        label=f"{label} {np.mean(data):.2f}/{np.std(data):.2f}",
        **kwargs,
    )


LOG_VARS = {
    "pt",
    "p",
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
    "pt": np.linspace(0, 2.5, 100),
    "p": np.linspace(0, 10, 100),
    "eta": np.linspace(-1, 5, 100),
    "theta": np.linspace(0, 0.8, 100),
    "phi": np.linspace(-np.pi, np.pi, 100),
    "beta": np.linspace(0.6, 1.1, 100),
    "vx": np.linspace(-2, 2, 50),
    "vy": np.linspace(-2, 2, 50),
    "vz": np.linspace(-20, 20, 50),
}

EVENT_BINS_DICT = {
    "ht": np.linspace(0, 7, 100),
    "ht_p": np.linspace(0, 25, 100),
    "Pxh": np.linspace(-4, 4, 100),
    "Pyh": np.linspace(-4, 4, 100),
    "met_x": np.linspace(-1.5, 1.5, 100),
    "met_y": np.linspace(-1.5, 1.5, 100),
    "Q2": np.linspace(0, 10, 100),
    "W": np.linspace(1.5, 4.5, 100),
    "xB": np.linspace(0, 1, 100),
    "npart": np.linspace(0, 25, 26) - 0.5,
    "Ng": np.linspace(0, 15, 15) - 0.5,
    "Nn": np.linspace(0, 15, 15) - 0.5,
}

EVENT_RES_BINS_DICT = {
    "ht": np.linspace(-2, 1, 50),
    "ht_p": np.linspace(-12, 12, 50),
    "Pxh": np.linspace(-4, 4, 50),
    "Pyh": np.linspace(-4, 4, 50),
    "beta": np.linspace(-0.1, 0.1, 50),
    "npart": np.linspace(-10, 10, 20) - 0.5,
    "met_x": np.linspace(-2.5, 2.5, 50),
    "met_y": np.linspace(-2.5, 2.5, 50),
    "Q2": np.linspace(-0.05, 0.05, 50),
    "xB": np.linspace(-0.05, 0.05, 50),
    "W": np.linspace(-0.1, 0.1, 50),
    "Ng": np.linspace(-5, 5, 10),
    "Nn": np.linspace(-5, 5, 10),
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
    "Q2": "$Q^2$",
    "W": "$W$",
    "npart": "$N_{part}$",
    "pt": "$p_T$",
    "p": "$p$",
    "eta": "$\\eta$",
    "theta": "$\\theta$",
    "phi": "$\\phi$",
    "vx": "$v_x$",
    "vy": "$v_y$",
    "vz": "$v_z$",
    "d0": "$d_0$",
    "z0": "$z_0$",
    "beta": "$\\beta$",
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
    def __init__(self, skip_jet_logs: bool = False, skip_met_logs: bool = False):
        super().__init__()
        self.validation_step_outputs = []
        self.is_setted_up = False
        self.skip_jet_logs = skip_jet_logs
        self.skip_met_logs = skip_met_logs

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
        # The plots skip diverged (non-finite) samples, so count them here to keep
        # the divergence visible.  Logged as a Python number: Lightning then puts
        # it on the module's device, and NCCL cannot all-reduce a CPU tensor.
        n_nonfinite = int((~torch.isfinite(fs_data)).flatten(1).any(dim=1).sum())
        pl_module.log(
            "val/fs_nonfinite",
            n_nonfinite,
            batch_size=batch_size,
            reduce_fx="sum",
            sync_dist=True,
        )
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

    def _save_and_log_figure(
        self,
        fig,
        plot_name: str,
        pl_module: LightningModule,
        image_prefix: str = "ED",
    ):
        """Save figure to file and log to experiment tracker.

        ``image_prefix`` is the Comet image name stem: figures logged in the same
        validation share a name and clobber each other in the tracker, so
        loggers that run alongside the default ``ED`` panel must pass a distinct
        prefix to keep their image separate.
        """
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.asarray(canvas.buffer_rgba())[..., :3]
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # Include global_step so step-based validation (val_check_interval) writes
        # one image per validation run instead of overwriting a single per-epoch
        # file. epoch is kept for readability; step guarantees uniqueness.
        tag = f"e{pl_module.current_epoch}_s{pl_module.global_step}"
        # Relative to the working directory, not the run directory; created on
        # first use so a fresh checkout does not fail at the first validation.
        fig_dir = Path("logs/figs")
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / f"{plot_name}_{tag}.png")
        if pl_module.logger is not None:
            pl_module.logger.experiment.log_image(
                image_data=image,
                name=f"{image_prefix}_{tag}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"{plot_name}_{tag}.png")
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

        for ax in ax_iter:
            ax.set_visible(False)

        for ax in axs.flatten():
            ax.set_box_aspect(1)
        self._save_and_log_figure(fig, plot_name, pl_module)
        return event_metrics

    def _prepare_plot_data(self, plot_dict):
        """Prepare data and determine bins/plot configuration."""
        # Only bin what this config actually generated: eta and theta both live in
        # PARTICLE_BINS_DICT, but any one config produces exactly one of them.
        bins_dict = {
            var: bins
            for var, bins in PARTICLE_BINS_DICT.items()
            if f"fs_{var}" in plot_dict or f"fs_{var}_shifted" in plot_dict
        }
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
        # if "fs_beta" in plot_dict:
        #     bins_dict["beta"] = PARTICLE_BINS_DICT["beta"]

        return plot_dict, bins_dict, plot_name

    def _create_figure(self, bins_dict):
        """Create figure with appropriate number of subplots."""
        # Particle vars + 1 class/charge panel from _plot_class_or_charge, plus the
        # event metrics: 1 class/pdgId + 2 jet (HT, HT_res) + 2 MET, each pair
        # dropped by its own flag.
        num_event_metrics = (
            1 + (0 if self.skip_jet_logs else 2) + (0 if self.skip_met_logs else 2)
        )
        num_plots = len(bins_dict) + 1 + num_event_metrics
        num_cols = 5 if self.skip_jet_logs else 6
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
        """Get mask for charged particles only (class < 3).

        For the PID config there is no ``class`` field, so the class is derived
        from ``pdgId`` via ``pdgid_to_class``.
        """
        if f"{target_collection}_class" in plot_dict:
            target_charged = plot_dict[f"{target_collection}_class"][fs_mask] < 3
            fs_charged = plot_dict["fs_class"][fs_mask] < 3
        else:
            target_charged = (
                pdgid_to_class(
                    torch.as_tensor(plot_dict[f"{target_collection}_pdgId"][fs_mask])
                ).numpy()
                < 3
            )
            fs_charged = (
                pdgid_to_class(torch.as_tensor(plot_dict["fs_pdgId"][fs_mask])).numpy()
                < 3
            )
        return target_charged & fs_charged

    def _plot_class_or_charge(self, plot_dict, ax, fs_mask):
        """Plot either charge histogram or class confusion matrix."""
        target_collection = self.target_collection

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
        elif "fs_class" in plot_dict:
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
            ax.set_xlabel("Fast Sim")
            ax.set_ylabel(target_collection.capitalize())
        else:
            ax.set_visible(False)

    @staticmethod
    def _get_pt(plot_dict, prefix):
        """Return pt for a collection, derived as p * sin(theta) when only p is stored."""
        if f"{prefix}_pt" in plot_dict:
            return plot_dict[f"{prefix}_pt"]
        return plot_dict[f"{prefix}_p"] * np.sin(plot_dict[f"{prefix}_theta"])

    def _plot_event_metrics(self, plot_dict, ax_iter, fs_mask):
        """Plot HT, MET, and class distributions. Returns computed metrics."""
        target_collection = self.target_collection
        ctxt_collection = self.ctxt_collection
        ctxt_mask = plot_dict["ctxt_mask"]

        # Compute event quantities.
        ctxt_pt = self._get_pt(plot_dict, ctxt_collection) * ctxt_mask
        ctxt_phi = plot_dict[f"{ctxt_collection}_phi"] * ctxt_mask
        ctxt_ht = ctxt_pt.sum(-1)

        target_pt = self._get_pt(plot_dict, target_collection) * fs_mask
        target_phi = plot_dict[f"{target_collection}_phi"] * fs_mask
        target_ht = target_pt.sum(-1)

        fs_pt = self._get_pt(plot_dict, "fs") * fs_mask
        fs_phi = plot_dict["fs_phi"] * fs_mask
        fs_ht = fs_pt.sum(-1)

        if not self.skip_met_logs:
            ctxt_met_x = (ctxt_pt * np.cos(ctxt_phi)).sum(-1)
            ctxt_met_y = (ctxt_pt * np.sin(ctxt_phi)).sum(-1)
            target_met_x = (target_pt * np.cos(target_phi)).sum(-1)
            target_met_y = (target_pt * np.sin(target_phi)).sum(-1)
            fs_met_x = (fs_pt * np.cos(fs_phi)).sum(-1)
            fs_met_y = (fs_pt * np.sin(fs_phi)).sum(-1)

        if not self.skip_jet_logs:
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

        if not self.skip_met_logs:
            # MET x
            met_x_bins = EVENT_BINS_DICT["met_x"]
            ax = next(ax_iter)
            _add_hist_1d(
                ax,
                target_met_x,
                met_x_bins,
                target_collection.capitalize(),
                histtype="stepfilled",
                alpha=0.5,
            )
            _add_hist_1d(ax, ctxt_met_x, met_x_bins, ctxt_collection.capitalize())
            _add_hist_1d(ax, fs_met_x, met_x_bins, "Fastsim")
            ax.set_title("MET x")
            ax.legend()

            # MET y
            met_y_bins = EVENT_BINS_DICT["met_y"]
            ax = next(ax_iter)
            _add_hist_1d(
                ax,
                target_met_y,
                met_y_bins,
                target_collection.capitalize(),
                histtype="stepfilled",
                alpha=0.5,
            )
            _add_hist_1d(ax, ctxt_met_y, met_y_bins, ctxt_collection.capitalize())
            _add_hist_1d(ax, fs_met_y, met_y_bins, "Fastsim")
            ax.set_title("MET y")
            ax.legend()

        # Class (or pdgId, when class is absent) distribution
        ax = next(ax_iter)
        if f"{target_collection}_class" in plot_dict:
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
        else:
            labels = [str(p) for p in pid_onehot_ids] + ["other"]
            pdg_bins = np.arange(len(labels) + 1) - 0.5
            target_idx = pdgid_to_onehot_index(
                torch.as_tensor(plot_dict[f"{target_collection}_pdgId"][fs_mask])
            ).numpy()
            fs_idx = pdgid_to_onehot_index(
                torch.as_tensor(plot_dict["fs_pdgId"][fs_mask])
            ).numpy()
            ax.hist(
                target_idx,
                bins=pdg_bins,
                histtype="stepfilled",
                label=target_collection.capitalize(),
                density=True,
                alpha=0.5,
            )
            ax.hist(
                fs_idx, bins=pdg_bins, histtype="step", label="Fastsim", density=True
            )
            ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
            ax.set_title("pdgId")
        ax.legend()

        metrics = {"ht": np.mean((fs_ht - target_ht) ** 2)}
        if not self.skip_met_logs:
            metrics["met_x"] = np.mean((fs_met_x - target_met_x) ** 2)
            metrics["met_y"] = np.mean((fs_met_y - target_met_y) ** 2)
        return metrics


class EventPerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        # Derive the targets from the stage's own fs_vars rather than a literal
        # list, so the callback needs no edit when fs_vars changes. get_base_name
        # (not scaler_key) is what plot_dict is keyed by; npflow -> "npart".
        event_vars = [
            self.unscaler.var_helper.get_base_name(v) for v in self.unscaler.fs_vars
        ]

        fig, axs = subplots_with_absolute_sized_axes(
            len(event_vars),
            2,
            figsize=(18, 4.5 * len(event_vars)),
            axis_width=8,
            axis_height=4,
            dpi=200,
        )
        target_collection = self.target_collection
        ctxt_collection = self.ctxt_collection

        for i, var in enumerate(event_vars):
            # Unfamiliar targets fall back to matplotlib auto-binning.
            bins = EVENT_BINS_DICT.get(var, 100)
            res_bins = EVENT_RES_BINS_DICT.get(var, 50)
            label = VARIABLES_LABELS.get(var, var)

            _add_hist_1d(
                axs[i][0],
                plot_dict[f"{target_collection}_{var}"],
                bins=bins,
                label=target_collection.capitalize(),
                histtype="stepfilled",
                alpha=0.5,
            )
            _add_hist_1d(
                axs[i][0],
                plot_dict[f"fs_{var}"],
                bins=bins,
                label="Fastsim",
                histtype="step",
            )
            _add_hist_1d(
                axs[i][0],
                plot_dict[f"{ctxt_collection}_{var}"],
                bins=bins,
                label=ctxt_collection.capitalize(),
                histtype="step",
                color="k",
            )
            axs[i][0].set_title(label)
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
                res_bins,
                f"{target_collection.capitalize()} - {ctxt_collection.capitalize()}",
                f"Fastsim - {ctxt_collection.capitalize()}",
            )
            axs[i][1].set_title(f"{label} Residuals")

        self._save_and_log_figure(fig, "EVT_ED", pl_module)
        pred_loss = {}
        for var in event_vars:
            sq_err = (
                plot_dict[f"fs_{var}"] - plot_dict[f"{target_collection}_{var}"]
            ) ** 2
            pred_loss[var] = np.mean(sq_err[np.isfinite(sq_err)])
        return pred_loss


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
