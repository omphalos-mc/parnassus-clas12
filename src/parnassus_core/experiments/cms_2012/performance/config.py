from parnassus_core.performance.config import BaseSample, PerformanceConfig


class CMS2012Sample(BaseSample):
    """Enumeration for the different samples used in the CMS 2012 performance evaluation."""

    TTBAR = "ttbar"
    QCD15TO3000 = "qcd15to3000"
    DYJETS = "dyjets"


class CMS2012PerformanceConfig(PerformanceConfig[CMS2012Sample]):
    sample_enum_cls = CMS2012Sample


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
