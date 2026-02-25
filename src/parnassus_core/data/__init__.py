"""Backend-agnostic physics dataset infrastructure.

This module provides:
- EagerDataset: Eager-loading dataset for ROOT files
- StreamingDataset: HDF5 dataset with on-the-fly loading
- ParnassusDataModule: LightningDataModule that sets up datasets and dataloaders based on file type
- RandomBatchSampler: Sampler for random batch sampling from streaming datasets
"""

from parnassus_core.data.dataset import EagerDataset
from parnassus_core.data.dataset_h5 import StreamingDataset
from parnassus_core.data.datamodule import ParnassusDataModule

__all__ = [
    "EagerDataset",
    "StreamingDataset",
    "ParnassusDataModule",
]
