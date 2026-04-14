from .bins import CMS2012BinsConfig
from .config import CMS2012Sample, CMS2012PerformanceConfig  # noqa: F401

from parnassus_core.performance import Performance, PlotHelper


class CMS2012Performance(Performance[CMS2012Sample]):
    pass


class CMS2012PlotHelper(PlotHelper[CMS2012Sample]):
    bins_config = CMS2012BinsConfig()  # type: ignore[assignment]
