from pathlib import Path
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

    if Path("/storage/agrp/dmitrykl/latex/texlive/2024/bin/x86_64-linux").exists():
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/bin/x86_64-linux"
        )

        # setup libgd
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/libgd-2.3.3/bin"
        )
        # setup dvipng
        os.environ["PATH"] = (
            os.environ["PATH"]
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/bin"
        )
        os.environ["LD_LIBRARY_PATH"] = (
            os.environ.get("LD_LIBRARY_PATH", "")
            + ":/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/libgd-2.3.3/lib:/storage/agrp/dmitrykl/latex/texlive/2024/texmf-dist/tex/latex/dvipng/kpathsea_aux/lib"
        )
    else:
        pass
