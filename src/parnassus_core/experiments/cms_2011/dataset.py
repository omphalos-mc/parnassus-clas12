import gc
from typing import Literal
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import uproot
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from parnassus_core.flow.utils.scaling import FeatureScaler

from lightning.pytorch import LightningDataModule
from .dataset_h5 import CMSDatasetHDF5, RandomBatchSampler

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


def normalize(inp):
    return np.arctan2(np.sin(inp), np.cos(inp))


def do_padding(tensor, max_len):
    shape = tensor.shape
    new_shape = (max_len,) + shape[1:]
    x = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    x[: shape[0]] = tensor
    return x


class CMSDataset(Dataset):
    def __init__(
        self,
        filename,
        max_particles: int,
        train_type: str,
        sin_cos: bool,
        zero_neutral_vtx: bool,
        scale_dict: str,
        var_dict: str,
        fs_mask_type: Literal["all", "charged", "neutral"] = "all",
        reduce_ds: float | int = 1.0,
        entry_start: int = 0,
        mode: Literal["train", "test", "eval"] = "train",
    ):
        super().__init__()
        self.feature_scaler = FeatureScaler(scale_dict)
        self.max_particles = max_particles
        with open(var_dict, "r") as f:
            self.variables_config = yaml.safe_load(f)
        self.truth_vars_to_load = self.variables_config["truth_vars_to_load"]
        self.pflow_vars_to_load = self.variables_config["pflow_vars_to_load"]
        self.fs_vars = self.variables_config["fs_vars"]
        self.ctxt_vars = self.variables_config["ctxt_vars"]
        self.ctxt_global_vars = self.variables_config["ctxt_global_vars"]
        self.fs_mask_type = fs_mask_type

        self.mode = mode

        assert self.mode in ["train", "test", "eval"], (
            "Mode must be either 'train', 'test' or 'eval'"
        )

        self.train_type = train_type
        self.sin_cos = sin_cos

        self.zero_neutral_vtx = zero_neutral_vtx

        self.file = uproot.open(filename, num_workers=6)
        self.tree = self.file["evt_tree"]

        self.entry_start = entry_start
        self.nevents = self.tree.num_entries
        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents * reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print(" we have ", self.nevents, " events")

        self.n_particle_mask = None
        self.full_data_array = {}

        self.n_truth_particles = self._load_data("truth", self.truth_vars_to_load)
        if self.mode != "eval":
            self.n_pflow_particles = self._load_data("pflow", self.pflow_vars_to_load)
        self._merge_and_preprocess(mask_events=True)

        if "pflow_d0" in self.full_data_array:
            for var in ["d0", "z0"]:
                self.full_data_array[f"pflow_{var}Sig"] = np.clip(
                    self.full_data_array[f"pflow_{var}"]
                    / self.full_data_array[f"pflow_{var}Error"],
                    -5e2,
                    5e2,
                )
        if (
            self.zero_neutral_vtx
            and self.mode != "eval"
            and "pflow_class" in self.full_data_array
        ):
            for var in ["vx", "vy", "vz"]:
                self.full_data_array[f"pflow_{var}"][
                    self.full_data_array["pflow_class"] > 2
                ] = 0
            if "pflow_d0" in self.full_data_array:
                for var in ["d0", "z0", "d0Error", "z0Error", "d0Sig", "z0Sig"]:
                    self.full_data_array[f"pflow_{var}"][
                        self.full_data_array["pflow_class"] > 2
                    ] = 0
                    self.full_data_array[f"pflow_{var}"][
                        self.full_data_array[f"pflow_{var}"] < -998
                    ] = 0

        self.full_data_array["eventNumber"] = torch.tensor(
            self.tree["eventNumber"].array(
                library="np",
                entry_stop=self.nevents + self.entry_start,
                entry_start=self.entry_start,
            )[self.n_particle_mask]
        )

        self.n_truth_particles = self.n_truth_particles[self.n_particle_mask]
        self.truth_cumsum = np.cumsum([0] + list(self.n_truth_particles))

        if self.mode != "eval":
            self.n_pflow_particles = self.n_pflow_particles[self.n_particle_mask]
            self.pflow_cumsum = np.cumsum([0] + list(self.n_pflow_particles))

        self._filter_vars()
        self.file.close()

        loaded_vars = set(self.full_data_array.keys())
        out_vars = set(self.fs_vars + self.ctxt_vars + self.ctxt_global_vars)
        for var in loaded_vars - out_vars:
            if var != "eventNumber":
                print(
                    f"Warning: Removing variable {var}, since it's not used in the model."
                )
                self.full_data_array.pop(var)

        self.nevents = len(self.n_truth_particles)
        del self.tree
        gc.collect()
        print("done loading data")

        self._get_scaled_ctxt_global_data()

    def _filter_vars(self):
        """Filter the variables, so we don't use pflow variables in the context in eval mode."""
        if self.mode == "eval":
            self.ctxt_global_vars = [
                var for var in self.ctxt_global_vars if not var.startswith("pflow_")
            ]
            self.ctxt_vars = [
                var for var in self.ctxt_vars if not var.startswith("pflow_")
            ]

    def _get_scaled_ctxt_global_data(self):
        self.scaled_ctxt_global_data = []
        means = None
        for var in self.ctxt_global_vars:
            var_name = var.replace("truth_", "").replace("pflow_", "")
            if var in self.full_data_array:
                self.scaled_ctxt_global_data.append(
                    self.feature_scaler[var_name].transform(self.full_data_array[var])
                )
            elif var == "ntruth":
                self.scaled_ctxt_global_data.append(
                    self.feature_scaler["npart"].transform(self.n_truth_particles)
                )
            elif var == "npflow" and self.mode != "eval":
                self.scaled_ctxt_global_data.append(
                    self.feature_scaler["npart"].transform(self.n_pflow_particles)
                )
            elif var == "means":
                means = self._calcualte_means()
        self.scaled_ctxt_global_data = np.stack(self.scaled_ctxt_global_data, axis=-1)
        if means is not None:
            self.scaled_ctxt_global_data = np.concatenate(
                [means, self.scaled_ctxt_global_data], axis=-1
            )

    def _merge_and_preprocess(self, mask_events=True):
        """
        Converts the data to torch tensors and applies the mask if needed
        """
        for var, value in tqdm(
            self.full_data_array.items(), desc="Merging and preprocessing data"
        ):
            if mask_events:
                value = value[self.n_particle_mask]
            if value.dtype == np.object_:
                value = np.concatenate(value)
            if "eta" in var:
                value = np.clip(value, -3, 3)
            elif "phi" in var:
                value = normalize(value)
            self.full_data_array[var] = value

    def _calcualte_means(self):
        n_vars = len(
            [
                el
                for el in self.ctxt_vars
                if (el.startswith("truth_") and "class" not in el)
            ]
        )
        means = np.zeros((self.nevents, n_vars), dtype=np.float32)
        for i, var in tqdm(enumerate(self.ctxt_vars), desc="Calculating means"):
            if var.startswith("truth_") and "class" not in var:
                means[:, i] = (
                    np.add.reduceat(
                        self.full_data_array[var],
                        self.truth_cumsum[:-1],
                    )
                    / self.n_truth_particles
                )
            # elif var.startswith("pflow_") and self.mode == "train":
            #     means[:, i] = (
            #         np.add.reduceat(
            #             self.full_data_array[var],
            #             self.pflow_cumsum[:-1],
            #         )
            #         / self.n_pflow_particles
            #     )
        return means

    def _load_data(self, collection: str, vars_to_load: list[str]):
        """
        Loads the data for the given collection and variables
        """
        n_particles = self.tree[f"n{collection}"].array(
            library="np",
            entry_stop=self.nevents + self.entry_start,
            entry_start=self.entry_start,
        )
        if self.n_particle_mask is None:
            self.n_particle_mask = (n_particles < self.max_particles) & (
                n_particles > 0
            )
        else:
            self.n_particle_mask = (
                self.n_particle_mask
                & (n_particles < self.max_particles)
                & (n_particles > 0)
            )

        arrays = self.tree.arrays(
            vars_to_load,
            library="np",
            entry_stop=self.nevents + self.entry_start,
            entry_start=self.entry_start,
        )
        for var in tqdm(vars_to_load, desc=f"Loading {collection} data"):
            self.full_data_array[var] = arrays[var]

        self.full_data_array[f"{collection}_ht"] = np.zeros(
            self.nevents, dtype=np.float32
        )
        self.full_data_array[f"{collection}_ptrel"] = []
        self.full_data_array[f"{collection}_met_x"] = np.zeros(
            self.nevents, dtype=np.float32
        )
        self.full_data_array[f"{collection}_met_y"] = np.zeros(
            self.nevents, dtype=np.float32
        )

        for i in range(self.nevents):
            self.full_data_array[f"{collection}_met_x"][i] = (
                self.full_data_array[f"{collection}_pt"][i]
                * np.cos(self.full_data_array[f"{collection}_phi"][i])
            ).sum()
            self.full_data_array[f"{collection}_met_y"][i] = (
                self.full_data_array[f"{collection}_pt"][i]
                * np.sin(self.full_data_array[f"{collection}_phi"][i])
            ).sum()
            self.full_data_array[f"{collection}_ht"][i] = self.full_data_array[
                f"{collection}_pt"
            ][i].sum()
            self.full_data_array[f"{collection}_ptrel"].append(
                self.full_data_array[f"{collection}_pt"][i]
                / self.full_data_array[f"{collection}_ht"][i]
            )
        self.full_data_array[f"{collection}_ptrel"] = np.array(
            self.full_data_array[f"{collection}_ptrel"], dtype=object
        )
        return n_particles

    def _get_ctxt_data(self, idx):
        """
        Returns the ctxt data for the given index

        Args:
            idx (int): index of the event

        Returns:
            ctxt_dict (dict): dictionary containing the ctxt data
        """
        n_truth_particles = self.n_truth_particles[idx]
        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]
        truth_idx = np.argsort(
            self.full_data_array["truth_ptrel"][truth_start:truth_end]
        )[::-1]

        if self.mode != "eval" and "pflow_ptrel" in self.full_data_array:
            n_pflow_particles = self.n_pflow_particles[idx]
            pflow_start, pflow_end = self.pflow_cumsum[idx], self.pflow_cumsum[idx + 1]
            pflow_idx = np.argsort(
                self.full_data_array["pflow_ptrel"][pflow_start:pflow_end]
            )[::-1]

        ctxt_data = []
        pf_ctxt_data = []
        for var in self.ctxt_vars:
            if "truth_" in var:
                x = self.full_data_array[var][truth_start:truth_end][truth_idx]
                arr_to_fill_ = ctxt_data
            elif "pflow_" in var and self.mode != "eval":
                x = self.full_data_array[var][pflow_start:pflow_end][pflow_idx]
                arr_to_fill_ = pf_ctxt_data
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
            if var.endswith("phi") and self.sin_cos:
                arr_to_fill_.append(torch.sin(x))
                arr_to_fill_.append(torch.cos(x))
            elif var.endswith("class"):
                arr_to_fill_.append(
                    F.one_hot(x.long().squeeze(), num_classes=5).float()
                )
            else:
                arr_to_fill_.append(
                    self.feature_scaler[
                        var.replace("truth_", "")
                        .replace("pflow_", "")
                        .replace("npflow", "npart")
                    ]
                    .transform(x)
                    .float()
                )
        ctxt_data = torch.cat(ctxt_data, dim=-1)
        ctxt_data = do_padding(ctxt_data, self.max_particles).float()
        ctxt_mask = torch.zeros(self.max_particles, dtype=torch.bool)
        ctxt_mask[:n_truth_particles] = True
        if len(pf_ctxt_data) > 0:
            pf_ctxt_data = torch.cat(pf_ctxt_data, dim=-1)
            pf_ctxt_data = do_padding(pf_ctxt_data, self.max_particles).float()
            pf_ctxt_mask = torch.zeros(self.max_particles, dtype=torch.bool)
            pf_ctxt_mask[:n_pflow_particles] = True
            return {
                "ctxt_data": ctxt_data,
                "ctxt_mask": ctxt_mask,
                "pf_ctxt_data": pf_ctxt_data,
                "pf_ctxt_mask": pf_ctxt_mask,
            }
        return {
            "ctxt_data": ctxt_data,
            "ctxt_mask": ctxt_mask,
        }

    def _get_fs_data(self, idx):
        """
        Returns the fs data for the given index

        Args:
            idx (int): index of the event

        Returns:
            fs_data (dict): dictionary containing the fs data
        """
        if self.train_type == "particle":
            n_pflow_particles = self.n_pflow_particles[idx]
            pflow_start, pflow_end = self.pflow_cumsum[idx], self.pflow_cumsum[idx + 1]
            pflow_idx = np.argsort(
                self.full_data_array["pflow_ptrel"][pflow_start:pflow_end]
            )[::-1]
        fs_data = []
        for var in self.fs_vars:
            if self.train_type == "particle":
                x = self.full_data_array[var][pflow_start:pflow_end][pflow_idx]
            elif var == "npflow":
                x = self.n_pflow_particles[idx]
            else:
                x = self.full_data_array[var][idx]
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
            if var.endswith("phi") and self.sin_cos:
                fs_data.append(torch.sin(x))
                fs_data.append(torch.cos(x))
            elif var.endswith("class"):
                fs_data.append(F.one_hot(x.long().squeeze(), num_classes=5).float())
            else:
                fs_data.append(
                    self.feature_scaler[
                        var.replace("truth_", "")
                        .replace("pflow_", "")
                        .replace("npflow", "npart")
                    ]
                    .transform(x)
                    .float()
                )
        fs_data = torch.cat(fs_data, dim=-1)
        fs_mask = torch.zeros(self.max_particles, dtype=torch.bool)

        if self.train_type == "particle":
            fs_data = do_padding(fs_data, self.max_particles)
            fs_mask[:n_pflow_particles] = True
            match self.fs_mask_type:
                case "all":
                    pass
                case "charged":
                    fs_mask[:n_pflow_particles] = fs_mask[:n_pflow_particles] & (
                        self.full_data_array["pflow_class"][pflow_start:pflow_end][
                            pflow_idx
                        ]
                        < 3
                    )
                case "neutral":
                    fs_mask[:n_pflow_particles] = fs_mask[:n_pflow_particles] & (
                        self.full_data_array["pflow_class"][pflow_start:pflow_end][
                            pflow_idx
                        ]
                        >= 3
                    )
        else:
            fs_data = fs_data.squeeze(0)

        return fs_data, fs_mask

    def __len__(self):
        return len(self.n_truth_particles)

    def __getitem__(self, idx):
        ctxt_dict = self._get_ctxt_data(idx)
        ctxt_dict["ctxt_global_data"] = torch.tensor(
            self.scaled_ctxt_global_data[idx]
        ).float()
        ctxt_dict["eventNumber"] = self.full_data_array["eventNumber"][idx].to(
            torch.int64
        )
        if self.mode != "eval":
            fs_data, fs_mask = self._get_fs_data(idx)
            ctxt_dict["fs_mask"] = fs_mask
            return fs_data, ctxt_dict
        return ctxt_dict


class CMSDataModule(LightningDataModule):
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
        num_test: int = -1,
        test_batch_size: int = 100,
        entry_start_train: int = 0,
        entry_start_valid: int = 0,
        entry_start_test: int = 0,
        eval_output_name: str | None = None,
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

        self.fs_mask_type = fs_mask_type

    def setup_dataset(self, path: str, num_events: int, entry_start: int, mode: str):
        args = {
            "filename": path,
            "max_particles": self.max_particles,
            "train_type": self.train_type,
            "sin_cos": self.sin_cos,
            "zero_neutral_vtx": self.zero_neutral_vtx,
            "scale_dict": self.scale_dict,
            "var_dict": self.var_dict,
            "fs_mask_type": self.fs_mask_type,
            "reduce_ds": num_events,
            "entry_start": entry_start,
            "mode": mode,
        }
        if path.endswith(".root"):
            return CMSDataset(**args)
        elif path.endswith(".h5"):
            return CMSDatasetHDF5(**args)
        raise ValueError(f"Expected file to end with .h5 or .root, got {path}")

    def setup(self, stage: str = None):
        if self.trainer.is_global_zero:
            print("-" * 100)

        # create training and validation datasets
        if stage in (None, "fit", "validate"):
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
        if stage in (None, "test"):
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
        dataset: CMSDataset | CMSDatasetHDF5,
        shuffle: bool,
        batch_size: int | None,
    ):
        print(f"Creating dataloader with {len(dataset):,} events")
        sampler = None
        if isinstance(dataset, CMSDatasetHDF5) and batch_size is not None:
            sampler = RandomBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                entry_start=dataset.entry_start,
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
            pin_memory=False,
        )

    def train_dataloader(self):
        print("Instantiating train dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(
            dataset=self.train_dset, shuffle=True, batch_size=self.train_batch_size
        )

    def val_dataloader(self):
        print("Instantiating validation dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(
            dataset=self.valid_dset, shuffle=False, batch_size=self.valid_batch_size
        )

    def test_dataloader(self):
        print("Instantiating test dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(
            dataset=self.test_dset, shuffle=False, batch_size=self.test_batch_size
        )
