from typing import ClassVar

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .bins import Clas12RGABinsConfig
from .config import Clas12RGASample, Clas12RGAPerformanceConfig  # noqa: F401
from .flat_data import Clas12FlatSampleData

from parnassus_core.performance import Performance, PlotHelper
from parnassus_core.performance.config import Source
from parnassus_core.performance.labels import SAMPLE_LABELS
from parnassus_core.utils.pid import pid_onehot_ids


class Clas12RGAPerformance(Performance[Clas12RGASample]):
    # Relabel the reference (pflow) collection for CLAS12 (not "CMS PFlow").
    reference_label: ClassVar[str] = "CLAS12 Recon"
    truth_label: ClassVar[str] = "Truth"

    # Fields read concurrently during load(). Each worker gets its own uproot handle;
    # see read_clas12_flat for why a shared one is unsafe. 1 (serial) is the default
    # because the win is modest and peak RSS roughly triples.
    n_workers: ClassVar[int] = 1

    def load(self):
        """Load via the flat-array backend instead of the shared object-array loader.

        ``process()`` is inherited unchanged -- ``Clas12FlatData`` overrides the same
        ``run_once``-decorated methods the base ``Data`` defines, so the driver in
        ``SampleData.process_data`` dispatches straight onto the vectorized versions.
        """
        for sample, config_list in self.config.sample_dict.items():
            self.sample_data_dict[sample] = Clas12FlatSampleData[Clas12RGASample].load(
                sample,
                config_list,
                reference_label=self.reference_label,
                truth_label=self.truth_label,
                n_workers=self.n_workers,
            )


# Display labels for the 12 CLAS12 PID one-hot classes; order tracks
# pid_onehot_ids ([22, 11, -11, 211, -211, 321, -321, 2212, -2212, 2112, 45])
# followed by the trailing "other" bucket.
PID_LABELS = [
    r"$\gamma$",
    r"$e^-$",
    r"$e^+$",
    r"$\pi^+$",
    r"$\pi^-$",
    r"$K^+$",
    r"$K^-$",
    r"$p$",
    r"$\bar{p}$",
    r"$n$",
    r"$d$",
    "other",
]


class Clas12RGAPlotHelper(PlotHelper[Clas12RGASample]):
    """CLAS12 flavour of the shared plot helper.

    On top of the variable-list overrides, every ``plot_features`` panel (event, DIS and
    particle grids, residuals included) is annotated with a chi2/ndf comparing each
    fastsim collection to the reference (``Source.PF``, "CLAS12 Recon"); see
    ``_annotate_panel``. ``plot_pdgId`` has its own self-contained path and is not
    annotated (confusion matrices, not 1D overlays).
    """

    bins_config: ClassVar[Clas12RGABinsConfig] = Clas12RGABinsConfig()

    event_vars: ClassVar[list[str]] = ["npart", "Pxh", "Pyh", "ht_p"]

    particle_vars: ClassVar[list[str]] = [
        "p",
        "theta",
        "phi",
        "beta",
        "vx",
        "vy",
        "vz",
    ]

    dis_vars: ClassVar[list[str]] = ["Q2", "xB", "W", "y"]

    # Collection (Data.name) every other collection is compared against for chi2/ndf.
    chi2_reference: ClassVar[str] = Source.PF.name

    def __init__(self, *args, show_chi2: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_chi2 = show_chi2
        # sample -> "<data_dict_label>/<var>" -> collection name -> (chi2, ndf).
        # Filled by _annotate_panel as plots are made, for printing/tabulating.
        self.chi2_values: dict[
            Clas12RGASample, dict[str, dict[str, tuple[float, int]]]
        ] = {}

    @staticmethod
    def _chi2_ndf(
        values: np.ndarray, ref_values: np.ndarray, bin_edges: np.ndarray
    ) -> tuple[float, int] | None:
        """Shape chi2 between two unbinned samples, with Poisson per-bin errors.

        Both samples are normalized to unit area over ``bin_edges`` (matching the
        ``density=True`` histograms that are drawn), so the number only probes shape, not
        the differing sample sizes:

            chi2 = sum_bins (p1 - p2)^2 / (sigma1^2 + sigma2^2),  sigma_i = sqrt(n_i)/N_i

        ndf is the number of contributing bins minus one for the unit-area constraint.
        Returns ``None`` when either sample has no entries inside the binning.
        """
        n1, _ = np.histogram(np.asarray(values, dtype=float), bins=bin_edges)
        n2, _ = np.histogram(np.asarray(ref_values, dtype=float), bins=bin_edges)
        n_tot_1, n_tot_2 = n1.sum(), n2.sum()
        if n_tot_1 == 0 or n_tot_2 == 0:
            return None
        p1, p2 = n1 / n_tot_1, n2 / n_tot_2
        variance = n1 / n_tot_1**2 + n2 / n_tot_2**2
        use = variance > 0
        if not use.any():
            return None
        chi2 = float(np.sum((p1 - p2)[use] ** 2 / variance[use]))
        return chi2, max(int(use.sum()) - 1, 1)

    def _annotate_panel(
        self,
        ax,
        sample: Clas12RGASample,
        var: str,
        data_dict_label: str,
        values: dict[str, np.ndarray],
        var_bins: np.ndarray | None,
    ) -> None:
        """Write chi2/ndf (vs. the reference collection) into the panel's upper right.

        Skipped when the panel was auto-binned (``var_bins is None``): each collection
        then has its own bin edges, so a two-sample chi2 is not defined.
        """
        if not self.show_chi2 or var_bins is None:
            return
        ref_values = values.get(self.chi2_reference)
        if ref_values is None:
            return
        data_dict = self.performance.sample_data_dict[sample].data_dict
        others = [name for name in values if name != self.chi2_reference]
        lines = []
        for name in others:
            result = self._chi2_ndf(values[name], ref_values, var_bins)
            if result is None:
                continue
            chi2, ndf = result
            var_key = f"{data_dict_label}/{var}"
            self.chi2_values.setdefault(sample, {}).setdefault(var_key, {})[name] = (
                chi2,
                ndf,
            )
            # Only name the collection when several are being compared.
            prefix = f"{data_dict[name].label}: " if len(others) > 1 else ""
            lines.append(rf"{prefix}$\chi^2$/ndf = {chi2 / ndf:.2f}")
        if not lines:
            return
        ax.text(
            0.97,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=max(self.font_size * 0.5, 8),
        )

    @staticmethod
    def _pdgid_to_index(pdgid):
        """Map pdgId values to the 12 PID one-hot indices (unknown -> 'other').

        One searchsorted pass over the (sorted) id table rather than one full-array
        comparison per id -- the flat arrays here hold ~18M entries.
        """
        arr = np.rint(np.asarray(pdgid)).astype(int)
        ids = np.asarray(pid_onehot_ids)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        pos = np.searchsorted(sorted_ids, arr)
        # Clip keeps the take in bounds for values above the largest id; the
        # equality check below then sends them to "other" anyway.
        pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        found = sorted_ids[pos_clipped] == arr
        return np.where(found, order[pos_clipped], len(ids))

    def plot_pdgId(self, normalize: str = "true"):
        """PID confusion matrix (reco vs truth pdgId) over the 12 CLAS12 one-hot
        classes: one row per sample, one column per non-truth collection, plus a
        trailing column with the overall PID-class fractions.

        Self-contained (does not touch the shared plot_classes / plot_cm path).
        Requires matched data (postprocess ``matching: true``) and ``pdgId`` in
        ``particle_data_all`` (loaded via the ``PARTICLE_VARS`` loader list).
        """
        n_classes = len(PID_LABELS)

        def matched_indices(pf_collection, tr_collection):
            """Truth/reco PID class of every matched pair, as two flat arrays.

            ``Clas12FlatData`` stores particles flat, so the matched pairs are one
            gather through the precomputed global indices rather than a loop over
            events.
            """
            pf_idx, tr_idx = pf_collection.global_match_indices(tr_collection)
            if len(pf_idx) == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            return (
                self._pdgid_to_index(tr_collection.particle_data_all["pdgId"][tr_idx]),
                self._pdgid_to_index(pf_collection.particle_data_all["pdgId"][pf_idx]),
            )

        num_cols = 0
        for sample in self.performance.sample_data_dict:
            sample_data = self.performance.sample_data_dict[sample]
            num_cols = max(
                num_cols, len(sample_data.data_dict) - 1 * (not self.plot_truth)
            )
        num_cols += 1  # overall histogram column
        fig, ax = plt.subplots(
            self.num_samples,
            num_cols,
            figsize=(num_cols * self.ax_width, self.num_samples * self.ax_height),
        )
        ax = np.asarray(ax)
        if ax.ndim == 1:
            ax = ax.reshape((self.num_samples, num_cols))

        for i, sample in enumerate(self.performance.sample_data_dict):
            sample_data = self.performance.sample_data_dict[sample]
            tr_data = sample_data.data_dict[Source.TR.name]
            col_idx = 0
            for pf_data in sample_data.data_dict.values():
                if pf_data.source == Source.TR:
                    continue
                data = pf_data.particle_data_all
                if "pdgId" not in data or "tr_idx" not in data:
                    continue
                tr_ix, pf_ix = matched_indices(pf_data, tr_data)
                if len(pf_ix) == 0:
                    continue
                cm = confusion_matrix(
                    tr_ix, pf_ix, labels=range(n_classes), normalize=normalize
                )
                sns.heatmap(
                    pd.DataFrame(cm, index=PID_LABELS, columns=PID_LABELS),
                    annot=True,
                    ax=ax[i, col_idx],
                    cmap=sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
                    fmt=".2f",
                    cbar=False,
                    annot_kws={"size": self.font_size // 3},
                )
                ax[i, col_idx].set_xlabel("Reco", fontsize=self.font_size)
                if i == 0:
                    ax[i, col_idx].set_title(pf_data.label, fontsize=self.font_size)
                if col_idx == 0:
                    ax[i, col_idx].set_ylabel(
                        SAMPLE_LABELS.get(sample.name.lower(), sample.name),
                        fontsize=self.font_size,
                    )
                col_idx += 1
                # Overall PID-class fractions for this collection.
                ax[i, -1].hist(
                    # Already flat (Clas12FlatData); no concatenate needed.
                    self._pdgid_to_index(data["pdgId"]),
                    bins=np.arange(n_classes + 1) - 0.5,
                    density=True,
                    histtype="stepfilled" if pf_data.source == Source.PF else "step",
                    label=pf_data.label,
                    **self.style_dict[pf_data.name],
                )
            if i == 0:
                ax[i, -1].set_title("Overall", fontsize=self.font_size)
            ax[i, -1].set_xticks(range(n_classes), PID_LABELS, rotation=90)
            ax[i, -1].legend(fancybox=True, loc="upper right")
            ax[i, -1].yaxis.tick_right()
            ax[i, -1].set_ylabel("a.u.", fontsize=self.font_size)
            ax[i, -1].yaxis.set_label_position("right")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.15)
        return fig
