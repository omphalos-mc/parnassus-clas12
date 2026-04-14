from typing import Literal

from torch.utils.data import DataLoader

from lightning.pytorch import LightningDataModule

from parnassus_core.data.dataset import EagerDataset
from parnassus_core.data.dataset_h5 import StreamingDataset
from parnassus_core.data.utils.sampler import RandomBatchSampler


class ParnassusDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        train_path: str,
        valid_path: str,
        test_path: str,
        num_train: int,
        num_valid: int,
        train_batch_size: int,
        valid_batch_size: int,
        max_particles: int,
        train_type: Literal["particle", "event"],
        sin_cos: bool,
        zero_neutral_vtx: bool,
        scale_dict: str,
        var_dict: str,
        fs_mask_type: Literal["all", "charged", "neutral"] = "all",
        ctxt_mask_type: Literal["all", "charged", "neutral"] = "all",
        recalculate_ip: bool = False,
        class_based_scaling: bool = False,
        num_test: int = -1,
        test_batch_size: int = 100,
        entry_start_train: int = 0,
        entry_start_valid: int = 0,
        entry_start_test: int = 0,
        eval_output_name: str | None = None,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.max_particles = max_particles
        self.train_type = train_type
        self.sin_cos = sin_cos
        self.zero_neutral_vtx = zero_neutral_vtx
        self.scale_dict = scale_dict
        self.var_dict = var_dict
        self.num_train = num_train
        self.num_valid = num_valid
        self.num_test = num_test
        self.entry_start_train = entry_start_train
        self.entry_start_valid = entry_start_valid
        self.entry_start_test = entry_start_test

        self.eval_output_name = eval_output_name
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

        self.fs_mask_type = fs_mask_type
        self.ctxt_mask_type = ctxt_mask_type
        self.recalculate_ip = recalculate_ip
        self.class_based_scaling = class_based_scaling

    def setup_dataset(self, path: str, num_events: int, entry_start: int, mode: str):
        print(f"Setting up dataset for {mode} with path: {path}")
        args = {
            "file_path": path,
            "max_particles": self.max_particles,
            "train_type": self.train_type,
            "sin_cos": self.sin_cos,
            "zero_neutral_vtx": self.zero_neutral_vtx,
            "scale_dict": self.scale_dict,
            "var_dict": self.var_dict,
            "fs_mask_type": self.fs_mask_type,
            "ctxt_mask_type": self.ctxt_mask_type,
            "reduce_ds": num_events,
            "entry_start": entry_start,
            "mode": mode,
            "recalculate_ip": self.recalculate_ip,
            "class_based_scaling": self.class_based_scaling,
        }
        if path.endswith(".root"):
            return EagerDataset(**args)
        elif path.endswith(".h5"):
            return StreamingDataset(**args)
        raise ValueError(f"Expected file to end with .h5 or .root, got {path}")

    def setup(self, stage: str):
        assert self.trainer is not None, "Trainer must be set before calling setup()"
        if self.trainer.is_global_zero:
            print("-" * 100)

        # create training and validation datasets
        if stage in ("fit", "validate"):
            print("Setting up training and validation datasets")
            self.train_dset = self.setup_dataset(
                path=self.train_path,
                num_events=self.num_train,
                entry_start=self.entry_start_train,
                mode="train",
            )
            self.valid_dset = self.setup_dataset(
                path=self.valid_path,
                num_events=self.num_valid,
                entry_start=self.entry_start_valid,
                mode="train",
            )
        if stage in ("test",):
            print("Setting up test dataset")
            self.test_dset = self.setup_dataset(
                path=self.test_path,
                num_events=self.num_test,
                entry_start=self.entry_start_test,
                mode="test",
            )

        # Only print train/val dataset details when actually training
        if self.trainer.is_global_zero:
            if stage in (None, "fit", "validate"):
                print(f"Created training dataset with {len(self.train_dset):,} events")
                print(
                    f"Created validation dataset with {len(self.valid_dset):,} events"
                )
            if stage in (None, "test"):
                print(f"Created test dataset with {len(self.test_dset):,} events")
            print("-" * 100, "\n")

    def get_dataloader(
        self,
        dataset: EagerDataset | StreamingDataset,
        shuffle: bool,
        batch_size: int | None,
    ):
        print(f"Creating dataloader with {len(dataset):,} events")
        sampler = None
        if isinstance(dataset, StreamingDataset) and batch_size is not None:
            sampler = RandomBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler already shuffles
            batch_size = None  # Sampler defines batch size
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=None,
            sampler=sampler,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context=self.multiprocessing_context,
        )

    def train_dataloader(self):
        assert self.trainer is not None, (
            "Trainer must be set before calling train_dataloader()"
        )
        print("Instantiating train dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(
            dataset=self.train_dset, shuffle=True, batch_size=self.train_batch_size
        )

    def val_dataloader(self):
        assert self.trainer is not None, (
            "Trainer must be set before calling val_dataloader()"
        )
        print("Instantiating validation dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(
            dataset=self.valid_dset, shuffle=False, batch_size=self.valid_batch_size
        )

    def test_dataloader(self):
        assert self.trainer is not None, (
            "Trainer must be set before calling test_dataloader()"
        )
        print("Instantiating test dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(
            dataset=self.test_dset, shuffle=False, batch_size=self.test_batch_size
        )
