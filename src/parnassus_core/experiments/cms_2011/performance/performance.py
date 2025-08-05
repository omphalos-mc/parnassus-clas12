from typing import ClassVar
from .bins import CMS2011BinsConfig
from .config import CMS2011Sample, CMS2011PerformanceConfig  # noqa: F401

from parnassus_core.performance import Performance, PlotHelper


class CMS2011Performance(Performance[CMS2011Sample]):
    pass


class CMS2011PlotHelper(PlotHelper[CMS2011Sample]):
    bins_config: ClassVar[CMS2011BinsConfig] = CMS2011BinsConfig()
