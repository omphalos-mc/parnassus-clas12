"""File loader backends for PhysicsDataset.

Provides abstract FileLoader interface and concrete implementations
for ROOT and HDF5 file formats.
"""

from parnassus_core.data.loaders.base import FileLoader
from parnassus_core.data.loaders.hdf5_loader import HDF5FileLoader
from parnassus_core.data.loaders.root_loader import ROOTFileLoader

__all__ = ["FileLoader", "HDF5FileLoader", "ROOTFileLoader"]
