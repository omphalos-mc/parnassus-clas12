from parnassus_core.performance.config import BaseSample, PerformanceConfig


class Clas12RGASample(BaseSample):
    """Enumeration for the different samples used in the CLAS12 RGA performance evaluation.

    CLAS12 RGA is a fixed-target electron-proton experiment, so (unlike the CMS
    experiments) there is currently a single physics sample. Add new members here as
    additional datasets are evaluated; the ``value`` must match the ``name`` used in the
    evaluation config / performance ``config_dict`` (lower-case is fine, the loader
    upper-cases the key when resolving the enum).
    """

    CLASDIS = "clasdis"
    CLASDIS_2 = "clasdis_2"
    CLASDIS_FILTERED = "clasdis_filtered"
    CLASDIS_PRECLEANED = "clasdis_precleaned"
    DVCS = "dvcs"
    PART_TEST = "part_test"
    PART_TEST_1 = "part_test_1"


class Clas12RGAPerformanceConfig(PerformanceConfig[Clas12RGASample]):
    sample_enum_cls = Clas12RGASample


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
