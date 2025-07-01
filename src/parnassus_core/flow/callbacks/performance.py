import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def normalize_phi(inp):
    return np.arctan2(np.sin(inp), np.cos(inp))


def _add_hist2d(fig, ax, pf_data, fs_data, bins, cmap):
    h, xe, ye, im = ax.hist2d(
        pf_data, fs_data, bins=bins, cmap=cmap, norm=mpl.colors.LogNorm()
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")


def _add_to_hist(ax, data, bins, label, histtype="step", **kwargs):
    ax.hist(
        data,
        bins=bins,
        histtype=histtype,
        label=f"{label} {np.mean(data):.2f}/{np.std(data):.2f}",
        **kwargs,
    )


BINS_2D_DICT = {
    "ptrel": np.linspace(0, 1, 100),
    "eta": np.linspace(-3, 3, 100),
    "phi": np.linspace(-np.pi, np.pi, 100),
    "vx": np.linspace(-2, 2, 50),
    "vy": np.linspace(-2, 2, 50),
    "vz": np.linspace(-20, 20, 50),
}

VARIABLES_LABELS = {
    "ht": "$H_T$",
    "met_x": "$MET_x$",
    "met_y": "$MET_y$",
    "npart": "$N_{part}$",
    "ptrel": "$p_{T, rel}$",
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

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pf_data, ctxt_dict = batch
        fs_mask = ctxt_dict["fs_mask"]
        return_dict = {}
        with torch.no_grad():
            fs_data = pl_module.sample(
                pf_data.shape,
                ctxt_dict=ctxt_dict,
            )
            # if pl_module.current_epoch % 5 == 0:
            #     pred_loss = pl_module.prediction_loss(fs_data, pf_data, fs_mask)
            #     for key, val in pred_loss.items():
            #         pl_module.log(
            #             f"pred_{key}",
            #             val,
            #             batch_size=pf_data.shape[0],
            #             sync_dist=True,
            #         )
        for key, val in ctxt_dict.items():
            return_dict[key] = val.cpu()
        return_dict["fs_data"] = fs_data.cpu()
        return_dict["pf_data"] = pf_data.cpu()

        self.validation_step_outputs.append(return_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        plot_dict = {}
        for el in self.validation_step_outputs:
            for key, val in el.items():
                if key not in plot_dict:
                    plot_dict[key] = []
                plot_dict[key].append(val)
        for key, val in plot_dict.items():
            if isinstance(val, list):
                plot_dict[key] = torch.cat(val, dim=0)
            else:
                plot_dict[key] = val
        batch_size = plot_dict["fs_data"].shape[0]
        plot_dict = self.unscale_variables(plot_dict)
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
            self.feature_scaler = valid_dataset.feature_scaler
            self.fs_vars = valid_dataset.fs_vars
            self.ctxt_vars = valid_dataset.ctxt_vars
            self.ctxt_global_vars = valid_dataset.ctxt_global_vars
            self.sin_cos = valid_dataset.sin_cos
            self.is_setted_up = True
        self.validation_step_outputs.clear()

    def unscale_variables(self, data_dict):
        new_data_dict = {}
        for i, var in enumerate(self.ctxt_global_vars):
            if var.startswith("truth_") or var.startswith("ntruth"):
                var = var.replace("truth_", "").replace("ntruth", "npart")
                new_data_dict[f"tr_{var}"] = (
                    self.feature_scaler[var]
                    .inverse_transform(data_dict["ctxt_global_data"][..., i])
                    .numpy()
                )
            elif var.startswith("pflow_"):
                var = var.replace("pflow_", "")
                new_data_dict[f"pf_{var}"] = (
                    self.feature_scaler[var]
                    .inverse_transform(data_dict["ctxt_global_data"][..., i])
                    .numpy()
                )
        var_idx = 0
        for var in self.ctxt_vars:
            if var.startswith("pflow_"):
                continue
            var_name = var.replace("truth_", "")
            if var_name.endswith("phi") and self.sin_cos:
                ctxt_ = torch.atan2(
                    data_dict["ctxt_data"][..., var_idx],
                    data_dict["ctxt_data"][..., var_idx + 1],
                )
                var_idx += 2
            elif var.endswith("class"):
                ctxt_ = data_dict["ctxt_data"][..., var_idx : var_idx + 5].argmax(-1)
                var_idx += 5
            else:
                ctxt_ = self.feature_scaler[var_name].inverse_transform(
                    data_dict["ctxt_data"][..., var_idx]
                )
                var_idx += 1
            new_data_dict[f"tr_{var_name}"] = ctxt_.numpy()
        var_idx = 0
        for var in self.fs_vars:
            var_name = var.replace("pflow_", "").replace("npflow", "npart")
            if var.endswith("phi") and self.sin_cos:
                fs_ = torch.atan2(
                    data_dict["fs_data"][..., var_idx],
                    data_dict["fs_data"][..., var_idx + 1],
                )
                pf_ = torch.atan2(
                    data_dict["pf_data"][..., var_idx],
                    data_dict["pf_data"][..., var_idx + 1],
                )
                var_idx += 2
            elif var.endswith("class"):
                fs_ = data_dict["fs_data"][..., var_idx : var_idx + 5].argmax(-1)
                pf_ = data_dict["pf_data"][..., var_idx : var_idx + 5].argmax(-1)
                var_idx += 5
            else:
                fs_ = self.feature_scaler[var_name].inverse_transform(
                    data_dict["fs_data"][..., var_idx]
                )
                pf_ = self.feature_scaler[var_name].inverse_transform(
                    data_dict["pf_data"][..., var_idx]
                )
                var_idx += 1
            new_data_dict[f"fs_{var_name}"] = fs_.numpy()
            new_data_dict[f"pf_{var_name}"] = pf_.numpy()
        if "ctxt_mask" in data_dict:
            new_data_dict["ctxt_mask"] = data_dict["ctxt_mask"].numpy()
        if "fs_mask" in data_dict:
            new_data_dict["fs_mask"] = data_dict["fs_mask"].numpy()
        return new_data_dict

    def plot_data(self, pl_module: LightningModule, plot_dict: dict[str, np.ndarray]):
        pass


class ParticlePerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        fig, axs = subplots_with_absolute_sized_axes(
            4, 3, figsize=(4 * 4, 20), axis_width=4, axis_height=4, dpi=200
        )
        axs = axs.flatten()
        canvas = FigureCanvas(fig)

        tr_mask = plot_dict["ctxt_mask"]
        pf_mask = plot_dict["fs_mask"].astype(bool)
        # pt eta phi plots

        cmap = copy.copy(plt.get_cmap("PuRd"))
        cmap.set_under("white")
        for i, (var, bins) in enumerate(BINS_2D_DICT.items()):
            axs[i].set_title(f"{VARIABLES_LABELS[var]} 2D")
            axs[i].set_xlabel("Pflow")
            axs[i].set_ylabel("FastSim")
            _add_hist2d(
                fig,
                axs[i],
                plot_dict[f"pf_{var}"][pf_mask],
                plot_dict[f"fs_{var}"][pf_mask],
                bins=bins,
                cmap=cmap,
            )
        i += 1
        cm = confusion_matrix(
            plot_dict["pf_class"][pf_mask],
            plot_dict["fs_class"][pf_mask],
            normalize="true",
        )
        df_cm = pd.DataFrame(
            cm,
            index=["Ch had", "El", "Mu", "Neut had", "Phot"],
            columns=["Ch had", "El", "Mu", "Neut had", "Phot"],
        )
        sns.heatmap(
            df_cm,
            annot=True,
            annot_kws={"size": 8},
            cmap=sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
            cbar=False,
            ax=axs[i],
        )
        axs[i].set_title("classes")
        axs[i].set_ylabel("Pflow")
        axs[i].set_xlabel("FastSim")
        i += 1

        tr_pt = plot_dict["tr_ptrel"] * tr_mask * plot_dict["tr_ht"].reshape(-1, 1)
        pf_pt = plot_dict["pf_ptrel"] * pf_mask * plot_dict["pf_ht"].reshape(-1, 1)
        fs_pt = plot_dict["fs_ptrel"] * pf_mask * plot_dict["pf_ht"].reshape(-1, 1)

        tr_phi = plot_dict["tr_phi"] * tr_mask
        pf_phi = plot_dict["pf_phi"] * pf_mask
        fs_phi = plot_dict["fs_phi"] * pf_mask

        tr_met_x = (tr_pt * np.cos(tr_phi)).sum(-1)
        tr_met_y = (tr_pt * np.sin(tr_phi)).sum(-1)
        pf_met_x = (pf_pt * np.cos(pf_phi)).sum(-1)
        pf_met_y = (pf_pt * np.sin(pf_phi)).sum(-1)
        fs_met_x = (fs_pt * np.cos(fs_phi)).sum(-1)
        fs_met_y = (fs_pt * np.sin(fs_phi)).sum(-1)

        tr_ht = tr_pt.sum(-1).flatten()
        pf_ht = pf_pt.sum(-1).flatten()
        fs_ht = fs_pt.sum(-1).flatten()
        bins = np.linspace(0, 3000, 100)

        _add_to_hist(axs[i], pf_ht, bins, "pf", histtype="stepfilled", alpha=0.5)
        _add_to_hist(axs[i], tr_ht, bins, "Truth")
        _add_to_hist(axs[i], fs_ht, bins, "FastSim")
        axs[i].set_title("Jet pT")
        axs[i].legend()

        bins = np.linspace(-200, 200, 100)
        _add_to_hist(
            axs[i + 1],
            pf_ht - tr_ht,
            bins,
            "pf",
            histtype="stepfilled",
            alpha=0.5,
        )
        _add_to_hist(axs[i + 1], fs_ht - tr_ht, bins, "FastSim")
        axs[i + 1].set_title("Jet pT Residual")
        axs[i + 1].legend()

        bins = np.linspace(-100, 100, 100)
        _add_to_hist(axs[i + 2], pf_met_x, bins, "pf", histtype="stepfilled", alpha=0.5)
        _add_to_hist(axs[i + 2], tr_met_x, bins, "Truth")
        _add_to_hist(axs[i + 2], fs_met_x, bins, "FastSim")
        axs[i + 2].set_title("MET x")
        axs[i + 2].legend()

        _add_to_hist(axs[i + 3], pf_met_y, bins, "pf", histtype="stepfilled", alpha=0.5)
        _add_to_hist(axs[i + 3], tr_met_y, bins, "Truth")
        _add_to_hist(axs[i + 3], fs_met_y, bins, "FastSim")
        axs[i + 3].set_title("MET y")
        axs[i + 3].legend()

        class_bins = np.linspace(-0.5, 4.5, 6)
        axs[i + 4].hist(
            plot_dict["pf_class"][pf_mask],
            bins=class_bins,
            histtype="stepfilled",
            label="pf",
            density=True,
            alpha=0.5,
        )
        axs[i + 4].hist(
            plot_dict["fs_class"][pf_mask],
            bins=class_bins,
            histtype="step",
            label="FastSim",
            density=True,
        )
        axs[i + 4].set_xticks(np.arange(5), ["Ch had", "El", "Mu", "Neut had", "Phot"])
        axs[i + 4].set_title("Class")
        axs[i + 4].legend()

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )
        for ax in axs:
            ax.set_box_aspect(1)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.savefig(f"ED_{pl_module.current_epoch}.png")
        if pl_module.logger is not None:
            pl_module.logger.experiment.log_image(
                image_data=image,
                name=f"ED_{pl_module.current_epoch}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"ED_{pl_module.current_epoch}.png")
        plt.close(fig)
        return {
            "ht": np.mean((fs_ht - pf_ht) ** 2),
            "met_x": np.mean((fs_met_x - pf_met_x) ** 2),
            "met_y": np.mean((fs_met_y - pf_met_y) ** 2),
        }


class EventPerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        fig, axs = subplots_with_absolute_sized_axes(
            4, 2, figsize=(18, 18), axis_width=8, axis_height=4, dpi=200
        )
        canvas = FigureCanvas(fig)

        cmap = copy.copy(plt.get_cmap("PuRd"))
        cmap.set_under("white")

        bins = {
            "ht": np.linspace(0, 3000, 100),
            "met_x": np.linspace(-100, 100, 100),
            "met_y": np.linspace(-100, 100, 100),
            "npart": np.linspace(0.5, 400.5, 100),
        }
        res_bins = {
            "ht": np.linspace(-2, 1, 50),
            "npart": np.linspace(-250, 150, 50),
            "met_x": np.linspace(-100, 100, 50),
            "met_y": np.linspace(-100, 100, 50),
        }
        for i, var in enumerate(["ht", "met_x", "met_y", "npart"]):
            _add_to_hist(
                axs[i][0],
                plot_dict[f"pf_{var}"],
                bins=bins[var],
                label=f"PF",
                histtype="stepfilled",
            )
            _add_to_hist(
                axs[i][0],
                plot_dict[f"fs_{var}"],
                bins=bins[var],
                label=f"FS",
                histtype="step",
            )
            _add_to_hist(
                axs[i][0],
                plot_dict[f"tr_{var}"],
                bins=bins[var],
                label=f"TR",
                histtype="step",
                color="k",
            )
            axs[i][0].set_title(f"{VARIABLES_LABELS[var]}")
            axs[i][0].legend()

            pf_res = (
                (plot_dict[f"pf_{var}"] - plot_dict[f"tr_{var}"])
                / plot_dict[f"pf_{var}"]
                if i == 0
                else plot_dict[f"pf_{var}"] - plot_dict[f"tr_{var}"]
            )
            fs_res = (
                (plot_dict[f"fs_{var}"] - plot_dict[f"tr_{var}"])
                / plot_dict[f"fs_{var}"]
                if i == 0
                else plot_dict[f"fs_{var}"] - plot_dict[f"tr_{var}"]
            )
            _add_to_hist(
                axs[i][1],
                pf_res,
                bins=res_bins[var],
                label=f"PF-TR",
                histtype="stepfilled",
            )
            _add_to_hist(
                axs[i][1], fs_res, bins=res_bins[var], label=f"FS-TR", histtype="step"
            )
            axs[i][1].set_title(f"{VARIABLES_LABELS[var]} Residuals")
            axs[i][1].legend()

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.savefig(f"ED_{pl_module.current_epoch}.png")
        if pl_module.logger is not None:
            pl_module.logger.experiment.log_image(
                image_data=image,
                name=f"ED_{pl_module.current_epoch}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"ED_{pl_module.current_epoch}.png")
        plt.close(fig)
        return {
            var: np.mean((plot_dict[f"fs_{var}"] - plot_dict[f"pf_{var}"]) ** 2)
            for var in ["ht", "met_x", "met_y", "npart"]
        }


class ImpactPerformanceLogger(PerformanceLogger):
    def plot_data(self, pl_module, plot_dict):
        fig, axs = subplots_with_absolute_sized_axes(
            2, 3, figsize=(20, 10), axis_width=6, axis_height=4, dpi=200
        )
        axs = axs.flatten()
        canvas = FigureCanvas(fig)

        cmap = copy.copy(plt.get_cmap("PuRd"))
        cmap.set_under("white")

        bins_dict = {
            "d0": np.linspace(-0.5, 0.5, 50),
            "d0Error": np.linspace(0, 1, 50),
            "d0Sig": np.linspace(-2, 2, 50),
            "z0": np.linspace(-0.5, 0.5, 50),
            "z0Error": np.linspace(0, 2, 50),
            "z0Sig": np.linspace(-2, 2, 50),
        }

        pf_mask = plot_dict["fs_mask"].astype(bool)
        for key, val in plot_dict.items():
            if (
                (key.startswith("pf_") or key.startswith("fs_"))
                and key != "fs_mask"
                and val.ndim == 2
            ):
                plot_dict[key] = val[pf_mask]
        for i, var in enumerate(bins_dict.keys()):
            if f"pf_{var}" not in plot_dict:
                if var in {"d0", "z0"}:
                    plot_dict[f"pf_{var}"] = (
                        plot_dict[f"pf_{var}Sig"] * plot_dict[f"pf_{var}Error"]
                    )
                    plot_dict[f"fs_{var}"] = (
                        plot_dict[f"fs_{var}Sig"] * plot_dict[f"fs_{var}Error"]
                    )
                elif var in {"d0Error", "z0Error"}:
                    var_ = var.replace("Error", "")
                    plot_dict[f"pf_{var}"] = (
                        plot_dict[f"pf_{var_}"] / plot_dict[f"pf_{var_}Sig"]
                    )
                    plot_dict[f"fs_{var}"] = (
                        plot_dict[f"fs_{var_}"] / plot_dict[f"fs_{var_}Sig"]
                    )
                elif var in {"d0Sig", "z0Sig"}:
                    var_ = var.replace("Sig", "")
                    plot_dict[f"pf_{var}"] = (
                        plot_dict[f"pf_{var_}"] / plot_dict[f"pf_{var_}Error"]
                    )
                    plot_dict[f"fs_{var}"] = (
                        plot_dict[f"fs_{var_}"] / plot_dict[f"fs_{var_}Error"]
                    )
            _add_to_hist(
                axs[i],
                plot_dict[f"pf_{var}"],
                bins=bins_dict[var],
                label="PF",
                histtype="stepfilled",
                color="blue",
                alpha=0.5,
            )
            _add_to_hist(
                axs[i],
                plot_dict[f"fs_{var}"],
                bins=bins_dict[var],
                label="FS",
                histtype="step",
                color="red",
            )
            axs[i].set_xlabel(VARIABLES_LABELS[var])
            axs[i].set_ylabel("Counts")
            axs[i].legend()
            axs[i].set_yscale("log")
            axs[i].set_title(var)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.savefig(f"ED_{pl_module.current_epoch}.png")
        if pl_module.logger is not None:
            pl_module.logger.experiment.log_image(
                image_data=image,
                name=f"ED_{pl_module.current_epoch}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"ED_{pl_module.current_epoch}.png")
        plt.close(fig)
        return {
            var: np.mean((plot_dict[f"fs_{var}"] - plot_dict[f"pf_{var}"]) ** 2)
            for var in ["d0", "d0Error", "d0Sig", "z0", "z0Error", "z0Sig"]
            if f"fs_{var}" in plot_dict
        }
