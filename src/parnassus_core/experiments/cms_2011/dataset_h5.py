from typing import Literal, cast

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from parnassus_core.flow.utils.scaling import FeatureScaler

TensorDict = dict[str, Tensor]

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


def normalize(inp):
    return np.arctan2(np.sin(inp), np.cos(inp))


def maybe_copy(src: np.ndarray) -> np.ndarray:
    """Return a contiguous array, copying only if needed.

    Parameters
    ----------
    src : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        ``src`` itself if it is already C-contiguous; otherwise a contiguous
        copy of ``src``.
    """
    if src.flags.c_contiguous:
        return src
    return src.copy()


class CMSDatasetHDF5(Dataset):
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

        # Configure file
        self.filename = filename
        self.file = h5py.File(self.filename)
        self.reduce_ds = reduce_ds
        self.entry_start = entry_start
        self.num = self.get_num_events()

        # Initialize all config stuff
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
        self._is_setup = False

        self.dss: dict[str, h5py.Dataset]
        self.arrays: dict[str, np.ndarray]

    def __del__(self):
        self.file.close()

    def __len__(self) -> int:
        return self.num

    def get_num_events(self) -> int:
        """Get the number of events in the dataset."""
        num_available = len(self.file["events"]) - self.entry_start
        if self.reduce_ds < 1.0 and self.reduce_ds > 0:
            num = int(num_available * self.reduce_ds)
        elif self.reduce_ds >= 1:
            num = min(int(self.reduce_ds), num_available)
        else:
            num = num_available
        return num

    def _setup(self):
        """Setup the dataset."""
        # setup datasets and accessor arrays
        self.dss: dict[str, h5py.Dataset] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.dss["truth"] = cast(h5py.Dataset, self.file["truth"])
        self.arrays["truth"] = np.array(0, dtype=self.dss["truth"].dtype)
        if self.mode != "eval":
            self.dss["pflow"] = cast(h5py.Dataset, self.file["pflow"])
            self.arrays["pflow"] = np.array(0, dtype=self.dss["pflow"].dtype)

        self.dss["events"] = cast(h5py.Dataset, self.file["events"])
        self.arrays["events"] = np.array(0, dtype=self.dss["events"].dtype)

        self._is_setup = True

    def _load_data(
        self, object_idx: slice, collection: str
    ) -> tuple[TensorDict, Tensor] | TensorDict:
        batch = self.arrays[collection]
        shape = (object_idx.stop - object_idx.start, *self.dss[collection].shape[1:])
        batch.resize(shape, refcheck=False)
        self.dss[collection].read_direct(batch, object_idx)

        data_dict: TensorDict = {}
        for var in batch.dtype.names:  # type: ignore[attr-defined]
            if var == "valid":
                continue
            # We don't care about 'pt' ordering, since its already sorted in preprocessing
            data_dict[var] = torch.from_numpy(maybe_copy(batch[var])).float()

        if "eta" in data_dict:
            data_dict["eta"] = torch.clip(data_dict["eta"], -3.0, 3.0)
        if "phi" in data_dict:
            data_dict["phi"] = torch.atan2(
                torch.sin(data_dict["phi"]), torch.cos(data_dict["phi"])
            )
        if collection == "events":
            return data_dict
        mask = torch.tensor(maybe_copy(batch["valid"]), dtype=torch.bool)
        data_dict["npart"] = torch.sum(mask, dim=1)
        return data_dict, mask

    def _calculate_means(
        self,
        truth_data: TensorDict,
    ) -> Tensor:
        """
        Calculate the means for the ctxt global data.

        Parameters
        ----------
            truth_data : TensorDict
                Dictionary containing truth-level data
        Returns
        -------
            means : Tensor
                Tensor containing the means
        """
        means = []
        for var in self.ctxt_vars:
            if var.startswith("truth_") and "class" not in var:
                means.append(
                    torch.nanmean(
                        truth_data[var.replace("truth_", "")], dim=1, keepdim=True
                    )
                )
        return torch.cat(means, dim=-1)

    def _get_ctxt_global_data(
        self,
        event_data: TensorDict,
        truth_data: TensorDict,
        pflow_data: TensorDict | None = None,
    ) -> Tensor:
        """
        Returns the ctxt global data.

        Parameters
        ----------
            event_data : TensorDict
                Dictionary containing event-level data
            truth_data : TensorDict
                Dictionary containing truth-level data
            pflow_data : TensorDict, optional
                Dictionary containing particle-flow data, by default None
        Returns
        -------
            ctxt_global_data : Tensor
                Tensor containing the ctxt global data
        """
        ctxt_global_data = []
        means = None
        for var in self.ctxt_global_vars:
            var_scale_name = var.replace("truth_", "").replace("pflow_", "")
            if var in event_data:
                ctxt_global_data.append(
                    self.feature_scaler[var_scale_name].transform(
                        event_data[var].view(-1, 1)
                    )
                )
            elif var == "npflow" and pflow_data is not None:
                x = pflow_data["npart"].float().view(-1, 1)
                ctxt_global_data.append(
                    self.feature_scaler["npart"].transform(x)
                )
            elif var == "ntruth":
                x = truth_data["npart"].float().view(-1, 1)
                ctxt_global_data.append(
                    self.feature_scaler["npart"].transform(x)
                )
            elif var == "means":
                means = self._calculate_means(truth_data)
        ctxt_global_data = torch.cat(ctxt_global_data, dim=-1)
        if means is not None:
            ctxt_global_data = torch.cat([means, ctxt_global_data], dim=-1)
        return ctxt_global_data

    def _get_ctxt_data(
        self,
        truth_data: TensorDict,
        truth_mask: Tensor,
        pflow_data: TensorDict | None = None,
        pflow_mask: Tensor | None = None,
    ) -> TensorDict:
        """
        Returns the ctxt data.

        Parameters
        ----------
            truth_data : TensorDict
                Dictionary containing truth-level data
            truth_mask : Tensor
                Mask tensor for truth-level data
            pflow_data : TensorDict, optional
                Dictionary containing particle-flow data, by default None
            pflow_mask : Tensor, optional
                Mask tensor for particle-flow data, by default None
        Returns
        -------
            ctxt_data : TensorDict
                Dictionary containing the ctxt data and masks

        """
        ctxt_data = []
        pf_ctxt_data = []
        for var in self.ctxt_vars:
            var_name = var.replace("truth_", "").replace("pflow_", "")
            if "truth_" in var:
                x = truth_data[var_name]
                arr_to_fill_ = ctxt_data
            elif "pflow_" in var and self.mode != "eval":
                if pflow_data is None:
                    raise ValueError(
                        "pflow_data must be provided when using pflow_ variables in ctxt_vars"
                    )
                x = pflow_data[var_name]
                arr_to_fill_ = pf_ctxt_data
            else:
                raise ValueError(
                    f"Variable {var} in ctxt_vars is not recognized. Must start with truth_ or pflow_"
                )
            x = x.float().unsqueeze(-1)
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
        if len(pf_ctxt_data) > 0:
            pf_ctxt_data = torch.cat(pf_ctxt_data, dim=-1)
            return {
                "ctxt_data": ctxt_data.nan_to_num(0.0),
                "ctxt_mask": truth_mask.bool(),
                "pf_ctxt_data": pf_ctxt_data.nan_to_num(0.0),
                "pf_ctxt_mask": pflow_mask.bool(),
            }
        return {
            "ctxt_data": ctxt_data.nan_to_num(0.0),
            "ctxt_mask": truth_mask.bool(),
        }

    def _get_fs_data(
        self, event_data: TensorDict, pflow_data: TensorDict, pflow_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns the fs data.

        Parameters
        ----------
            event_data : TensorDict
                Dictionary containing event-level data
            pflow_data : TensorDict
                Dictionary containing particle-flow data
            pflow_mask : Tensor
                Mask tensor for particle-flow data
        Returns
        -------
            fs_data : Tensor
                Tensor containing the fs data
            fs_mask : Tensor
                Mask tensor for fs data
        """
        fs_data = []
        for var in self.fs_vars:
            if self.train_type == "particle":
                x = pflow_data[var.replace("pflow_", "")]
            elif var == "npflow":
                x = pflow_data["npart"]
            else:
                x = event_data[var]
            x = x.float().unsqueeze(-1)
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
        fs_mask = pflow_mask.clone()
        if self.train_type == "particle":
            match self.fs_mask_type:
                case "all":
                    pass
                case "charged":
                    fs_mask = fs_mask & (pflow_data["class"] < 3)
                case "neutral":
                    fs_mask = fs_mask & (pflow_data["class"] >= 3)
        else:
            fs_data = fs_data.squeeze(0)

        return fs_data, fs_mask

    def __getitem__(self, object_idx: slice) -> tuple[Tensor, TensorDict] | TensorDict:
        if not self._is_setup:
            self._setup()

        event_data = self._load_data(
            object_idx=object_idx,
            collection="events",
        )
        event_data = cast(TensorDict, event_data)

        truth_data, truth_mask = self._load_data(
            object_idx=object_idx,
            collection="truth",
        )
        # Need to cast for type checker
        truth_data = cast(TensorDict, truth_data)
        truth_mask = cast(Tensor, truth_mask)
        if "pt" in truth_data:
            truth_data["ptrel"] = truth_data["pt"] / event_data["truth_ht"].view(-1, 1)

        if self.mode != "eval":
            pflow_data, pflow_mask = self._load_data(
                object_idx=object_idx,
                collection="pflow",
            )
            # Need to cast for type checker
            pflow_data = cast(TensorDict, pflow_data)
            pflow_mask = cast(Tensor, pflow_mask)

            if "pt" in pflow_data:
                pflow_data["ptrel"] = pflow_data["pt"] / event_data["pflow_ht"].view(
                    -1, 1
                )
        else:
            pflow_data = None
            pflow_mask = None

        ctxt_dict = self._get_ctxt_data(
            truth_data=truth_data,
            truth_mask=truth_mask,
            pflow_data=pflow_data,
            pflow_mask=pflow_mask,
        )
        ctxt_dict["ctxt_global_data"] = self._get_ctxt_global_data(
            event_data=event_data,
            truth_data=truth_data,
            pflow_data=pflow_data,
        ).nan_to_num(0.0)
        ctxt_dict["eventNumber"] = event_data["eventNumber"]

        if self.mode != "eval":
            fs_data, fs_mask = self._get_fs_data(
                event_data=event_data,
                pflow_data=pflow_data,
                pflow_mask=pflow_mask,
            )
            fs_data.nan_to_num_(0.0)
            ctxt_dict["fs_mask"] = fs_mask
            return fs_data, ctxt_dict
        return ctxt_dict


class RandomBatchSampler(Sampler):
    """Batch sampler for an h5 dataset.

    The batch sampler performs weak shuffling. Objects are batched first,
    and then batches are shuffled.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Input dataset
    batch_size : int
        Number of objects to batch
    entry_start: int
        Starting index.
    shuffle : bool, optional
        Shuffle the batches, by default False
    drop_last : bool, optional
        Drop the last incomplete batch (if present), by default False
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        entry_start: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.entry_start = entry_start
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.nonzero_last_batch = int(self.n_batches) < self.n_batches
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        return int(self.n_batches) + int(not self.drop_last and self.nonzero_last_batch)

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(int(self.n_batches))
        else:
            self.batch_ids = torch.arange(int(self.n_batches))
        # yield full batches from the dataset
        for batch_id in self.batch_ids:
            start, stop = (
                self.entry_start + batch_id * self.batch_size,
                self.entry_start + (batch_id + 1) * self.batch_size,
            )
            yield np.s_[int(start) : int(stop)]

        # in case the batch size is not a perfect multiple of the number of samples,
        # yield the remaining samples
        if not self.drop_last and self.nonzero_last_batch:
            start, stop = (
                self.entry_start + int(self.n_batches) * self.batch_size,
                self.entry_start + self.dataset_length,
            )
            yield np.s_[int(start) : int(stop)]
