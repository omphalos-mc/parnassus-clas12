import math
from collections import defaultdict
from typing import ClassVar, Generic, Literal, Type, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


from .labels import VAR_LABELS, RESIDUAL_VAR_LABELS, SAMPLE_LABELS
from .bins import BinsConfig
from .config import SampleT, Source
from .utils import get_classes
from .performance import Performance

EVENT_VARS = ["npart", "njet", "met_x", "met_y", "ht"]

# DIS kinematics loaded into event_data by postprocess (cfg.compute_dis). The bin
# edges live in each experiment's BinsConfig (dis_bins/dis_res_bins).
DIS_VARS = ["Q2", "xB", "W", "y", "nu"]

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


def plot_cm(
    ax,
    pf_class: np.ndarray,
    tr_class: np.ndarray,
    normalize: Literal["true", "pred", "all"] = "true",
    annot: bool = True,
    xticklables="auto",
    yticklables="auto",
    annot_fs=16,
):
    class_labels = ["Ch. had", r"$e^{\pm}$", r"${\mu}^{\pm}$", "Nu. had", r"$\gamma$"]
    cm_pf = confusion_matrix(tr_class, pf_class, labels=range(5), normalize=normalize)
    df_cm = pd.DataFrame(cm_pf, index=class_labels, columns=class_labels)
    sns.heatmap(
        df_cm,
        annot=annot,
        ax=ax,
        cmap=sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
        fmt=".2f",
        cbar=False,
        annot_kws={"size": annot_fs},
        xticklabels=xticklables,
        yticklabels=yticklables,
    )


class PlotHelper(Generic[SampleT]):
    bins_config: ClassVar[Type[BinsConfig[SampleT]]]  # type: ignore[misc]

    # Variable lists driving each plot_*_features grid. Default to the module
    # constants (CMS behavior); override per experiment to add/drop variables.
    # e.g. CLAS12 drops "njet" from event_vars because jets are not clustered.
    event_vars: ClassVar[list[str]] = EVENT_VARS
    jet_vars: ClassVar[list[str]] = JET_VARS
    particle_vars: ClassVar[list[str]] = PARTICLE_VARS
    dis_vars: ClassVar[list[str]] = DIS_VARS

    def __init__(
        self,
        performance: Performance,
        plot_truth: bool = False,
        font_size: int = FONT_SIZE,
        ax_height: int = AX_HEIGHT,
        ax_width: int = AX_WIDTH,
        style_dict: dict | None = None,
        drop_empty_events: bool = False,
        ncols: int | None = None,
    ):
        self.performance = performance
        self.plot_truth = plot_truth
        self.drop_empty_events = drop_empty_events
        # Default panel columns for the plot_features grids (None: one per variable).
        self.ncols = ncols
        self.num_samples = len(performance.sample_data_dict)

        self.font_size = font_size
        self.ax_height = ax_height
        self.ax_width = ax_width
        self.style_dict = style_dict if style_dict is not None else DEFAULT_STYLE_DICT

    def plot_features(
        self,
        data_dict_label: str,
        var_list: list[str],
        bins: dict[str, dict[SampleT, np.ndarray]] | None = None,
        log_scale_vars: list[str] | str = [],
        xlabel_prefix: str = "",
        particle_mask_type: str = "all",
        drop_empty_events: bool | None = None,
        ncols: int | None = None,
    ):
        # Per-collection: each drops rows where *its own* npart == 0. A common mask
        # across collections would leave only their intersection, which for a
        # union-join file silently reverts to the intersect-join comparison.
        drop_empty = (
            self.drop_empty_events if drop_empty_events is None else drop_empty_events
        )
        var_labels = VAR_LABELS | RESIDUAL_VAR_LABELS
        num_legends = 0
        # Each sample is a block of n_rows grid rows, its variables filled row by row
        # into ncols columns. Default: one row per sample, one column per variable.
        ncols = min(ncols or self.ncols or len(var_list), len(var_list))
        n_rows = math.ceil(len(var_list) / ncols)
        fig, ax = plt.subplots(
            self.num_samples * n_rows,
            ncols,
            figsize=(ncols * self.ax_width, self.num_samples * n_rows * self.ax_height),
            squeeze=False,
        )
        ax = cast(np.ndarray, ax)
        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            num_legends = len(sample_data.data_dict) - 1 * (not self.plot_truth)
            block = ax[i * n_rows : (i + 1) * n_rows].ravel()
            for unused in block[len(var_list) :]:
                unused.set_visible(False)
            for j, var in enumerate(var_list):
                panel = block[j]
                var_bins = None if bins is None else bins[var][sample]
                # Values actually histogrammed in this panel, keyed by collection
                # name -- handed to _annotate_panel() below so subclasses can add
                # per-panel numbers without recomputing the masks/transforms.
                panel_values: dict[str, np.ndarray] = {}
                for pf_data in sample_data.data_dict.values():
                    if not self.plot_truth and (pf_data.source == Source.TR):
                        continue
                    data_dict = getattr(pf_data, data_dict_label)
                    value = data_dict[var]
                    # Restricted to event_data on purpose: in particle_data_* the
                    # "_res" columns are flat arrays of matched *particles*, not one
                    # entry per event, so an event-index mask would mis-slice them.
                    # (Masking particle data would be a no-op anyway -- an empty event
                    # contributes no entries to the concatenate below.)
                    if drop_empty and data_dict_label == "event_data":
                        npart = pf_data.event_data.get("npart")
                        if npart is not None and len(npart) == len(value):
                            value = np.asarray(value)[npart > 0]
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
                    panel_values[pf_data.name] = value
                    panel.hist(
                        value,
                        bins=var_bins,
                        density=True,
                        histtype="stepfilled"
                        if pf_data.source == Source.PF
                        else "step",
                        label=pf_data.label,
                        **self.style_dict[pf_data.name],
                    )

                self._annotate_panel(
                    panel, sample, var, data_dict_label, panel_values, var_bins
                )

                panel.yaxis.tick_right()
                if var in log_scale_vars:
                    panel.set_yscale("log")
                if i == self.num_samples - 1:
                    panel.set_xlabel(
                        xlabel_prefix + var_labels[var], fontsize=self.font_size
                    )
            # Axes fraction of the block's first column, so the label is centered on
            # all of the sample's rows (a single panel when the grid is not wrapped).
            first_col = ax[i * n_rows : (i + 1) * n_rows, 0]
            block[0].annotate(
                # Fall back to the enum name for samples without a curated label
                # (same guard as plot_pdgId).
                SAMPLE_LABELS.get(sample.name.lower(), sample.name),
                xy=(-0.1, 0.5),
                xycoords=lambda renderer, axes=first_col: Bbox.union(
                    [a.get_window_extent(renderer) for a in axes]
                ),
                fontsize=self.font_size,
                ha="center",
                va="center",
                rotation=90,
            )
            # "a.u." on the right-most used panel of each row
            for row_start in range(0, len(var_list), ncols):
                row_end = block[min(row_start + ncols, len(var_list)) - 1]
                row_end.set_ylabel("a.u.", fontsize=self.font_size)
                row_end.yaxis.set_label_position("right")

        # For an even ncols, start the legend on the gap right of this panel (the grid
        # center), so tight_layout does not reserve room for an off-center overhang.
        leg = ax[0, (ncols - 1) // 2].legend(
            loc="upper center",
            ncol=num_legends,
            fancybox=True,
            fontsize=self.font_size if n_rows == 1 else 0.8 * self.font_size,
            bbox_to_anchor=(0.5 if ncols % 2 else 1.075, 1.325),
        )
        fig.tight_layout()
        # Wrapped rows carry x labels between them and need the extra height.
        fig.subplots_adjust(hspace=0.1 if n_rows == 1 else 0.3, wspace=0.15)
        if n_rows > 1:
            # A single-row grid fits the legend band, a panel and the x-label band in
            # ax_height; a wrapped one has the bands only once. Rescale the height so
            # the panels keep the single-row aspect ratio.
            W, H = fig.get_size_inches()
            top, bottom = ax[0, 0].get_position(), ax[-1, 0].get_position()
            bands = H * (1 - top.y1 + bottom.y0)
            scale = (self.ax_height - bands) / (top.height * H)
            fig.set_size_inches(W, bands + (H - bands) * scale)
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.3, wspace=0.15)
        # An axes anchor centers the legend on one panel, which is the middle of the grid
        # only for an odd number of columns. Re-anchor to the grid center (same y).
        first, last = ax[0, 0].get_position(), ax[0, -1].get_position()
        leg.set_bbox_to_anchor(
            ((first.x0 + last.x1) / 2, first.y0 + 1.325 * first.height),
            transform=fig.transFigure,
        )
        return fig

    def _annotate_panel(
        self,
        ax,
        sample: SampleT,
        var: str,
        data_dict_label: str,
        values: dict[str, np.ndarray],
        var_bins: np.ndarray | None,
    ) -> None:
        """Hook called once per ``plot_features`` panel, after its histograms are drawn.

        No-op by default. ``values`` maps collection name (``Data.name``, e.g. "PF" or a
        fastsim entry name) to the exact array that was histogrammed, and ``var_bins`` is
        the shared bin edges (``None`` when matplotlib auto-binned each collection).
        Override per experiment to add per-panel numbers -- see
        ``Clas12RGAPlotHelper._annotate_panel`` for the CLAS12 chi2/ndf.
        """
        return None

    def plot_classes(self):
        num_cols = 0
        for sample in self.performance.sample_data_dict:
            sample_data = self.performance.sample_data_dict[sample]
            num_cols = max(
                num_cols, len(sample_data.data_dict) - 1 * (not self.plot_truth)
            )
        num_cols += 1  # For overall histogram
        fig, ax = plt.subplots(
            self.num_samples,
            num_cols,
            figsize=(num_cols * self.ax_width, self.num_samples * self.ax_height),
        )
        ax = cast(np.ndarray, ax)
        if len(ax.shape) == 1:
            ax = ax.reshape((self.num_samples, num_cols))
        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            col_idx = 0
            tr_data_dict = getattr(
                sample_data.data_dict[Source.TR.name], "particle_data_all"
            )
            for pf_data in sample_data.data_dict.values():
                if pf_data.source == Source.TR:
                    continue
                data_dict: dict[str, np.ndarray] = getattr(pf_data, "particle_data_all")
                pf_class_matched, tr_class_matched = get_classes(
                    data_dict["tr_idx"], data_dict["pf_idx"], data_dict, tr_data_dict
                )
                plot_cm(
                    ax[i, col_idx],
                    pf_class_matched,
                    tr_class_matched,
                    normalize="true",
                    annot=True,
                    xticklables="auto",
                    yticklables="auto",
                    annot_fs=self.font_size // 2,
                )
                if i == 0:
                    ax[i, col_idx].set_title(pf_data.label, fontsize=self.font_size)
                if col_idx == 0:
                    ax[i, col_idx].set_ylabel(
                        SAMPLE_LABELS[sample.name.lower()], fontsize=self.font_size
                    )
                col_idx += 1
                # Overall histogram
                ax[i, -1].hist(
                    np.concatenate(data_dict["class"]),
                    bins=np.arange(6) - 0.5,
                    density=True,
                    histtype="stepfilled" if pf_data.source == Source.PF else "step",
                    label=pf_data.label,
                    **self.style_dict[pf_data.name],
                )
            if i == 0:
                ax[i, -1].set_title("Overall", fontsize=self.font_size)
            ax[i, -1].set_xticks(
                range(5),
                ["Ch. had", r"$e^{\pm}$", r"${\mu}^{\pm}$", "Nu. had", r"$\gamma$"],
            )
            ax[i, -1].legend(
                fancybox=True,
                loc="upper right",
            )
            ax[i, -1].yaxis.tick_right()
            ax[i, -1].set_ylabel("a.u.", fontsize=self.font_size)
            ax[i, -1].yaxis.set_label_position("right")
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.15)
        return fig

    def plot_event_features(
        self,
        bins: dict[str, dict[SampleT, np.ndarray]] | None = None,
        residuals: bool = False,
        drop_empty_events: bool | None = None,
        ncols: int | None = None,
    ):
        if residuals:
            bins = self.bins_config.event_res_bins if bins is None else bins  # type: ignore[assignment]
            event_vars = [f"{var}_res" for var in self.event_vars]
        else:
            bins = self.bins_config.event_bins if bins is None else bins  # type: ignore[assignment]
            event_vars = self.event_vars
        fig = self.plot_features(
            "event_data",
            event_vars,
            bins=bins,
            drop_empty_events=drop_empty_events,
            ncols=ncols,
        )
        return fig

    def plot_dis_features(
        self,
        bins: dict[str, dict[SampleT, np.ndarray]] | None = None,
        residuals: bool = False,
        drop_empty_events: bool | None = None,
        ncols: int | None = None,
    ):
        # DIS variables live in event_data (loaded from postprocess, cfg.compute_dis).
        # Bin edges come from the experiment's BinsConfig.dis_bins/dis_res_bins.
        if residuals:
            bins = self.bins_config.dis_res_bins if bins is None else bins  # type: ignore[assignment]
            dis_vars = [f"{var}_res" for var in self.dis_vars]
        else:
            bins = self.bins_config.dis_bins if bins is None else bins  # type: ignore[assignment]
            dis_vars = self.dis_vars
        fig = self.plot_features(
            "event_data",
            dis_vars,
            bins=bins,
            drop_empty_events=drop_empty_events,
            ncols=ncols,
        )
        return fig

    def plot_jet_features(
        self,
        bins: dict[str, dict[SampleT, np.ndarray]] | None = None,
        residuals: bool = False,
        log_scale_vars: list[str] | str = [],
        ncols: int | None = None,
    ):
        if residuals:
            bins = self.bins_config.jet_res_bins if bins is None else bins  # type: ignore[assignment]
            jet_vars = [f"{var}_res" for var in self.jet_vars]
        else:
            bins = self.bins_config.jet_bins if bins is None else bins  # type: ignore[assignment]
            jet_vars = self.jet_vars
        if log_scale_vars == "all":
            log_scale_vars = jet_vars
        xlabel_prefix = "Residual jet " if residuals else "Jet "
        fig = self.plot_features(
            "jet_data",
            jet_vars,
            bins=bins,
            log_scale_vars=log_scale_vars,
            xlabel_prefix=xlabel_prefix,
            ncols=ncols,
        )
        return fig

    def plot_particle_features(
        self,
        bins: dict[str, dict[SampleT, np.ndarray]] | None = None,
        collection: str = "all",
        residuals: bool = False,
        log_scale_vars: list[str] | str = [],
        particle_mask_type: str = "all",
        ncols: int | None = None,
    ):
        if residuals:
            bins = self.bins_config.particle_res_bins if bins is None else bins  # type: ignore[assignment]
            particle_vars = [f"{var}_res" for var in self.particle_vars]
        else:
            bins = self.bins_config.particle_bins if bins is None else bins  # type: ignore[assignment]
            particle_vars = self.particle_vars
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
            ncols=ncols,
        )
        return fig

    def plot_impact_features(
        self,
        bins: dict[str, dict[SampleT, np.ndarray]] | None = None,
        collection: str = "all",
        log_scale_vars: list[str] | str = [],
        particle_mask_type: Literal["had", "el", "mu", "charged"] = "charged",
        ncols: int | None = None,
    ):
        bins = self.bins_config.particle_bins if bins is None else bins  # type: ignore[assignment]
        impact_vars = IMPACT_VARS
        if log_scale_vars == "all":
            log_scale_vars = impact_vars
        fig = self.plot_features(
            f"particle_data_{collection}",
            impact_vars,
            bins=bins,
            log_scale_vars=log_scale_vars,
            particle_mask_type=particle_mask_type,
            ncols=ncols,
        )
        return fig
