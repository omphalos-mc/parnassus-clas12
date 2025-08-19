from parnassus_core.performance.config import BaseSample, PerformanceConfig


class CMS2011Sample(BaseSample):
    """Enumeration for the different samples used in the CMS 2011 performance evaluation."""

    H4LEP = "h4lep"
    TTBAR = "ttbar"
    QCD_470 = "qcd_470"
    QCD_600 = "qcd_600"


class CMS2011PerformanceConfig(PerformanceConfig[CMS2011Sample]):
    sample_enum_cls = CMS2011Sample


def init_latex():
    """Initialize the LaTeX environment for the plotting."""
    import os
    import matplotlib as mpl

    if "tex" in os.environ.get("PATH", ""):
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        pass
