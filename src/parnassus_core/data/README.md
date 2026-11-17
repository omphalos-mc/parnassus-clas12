# `parnassus_core.data`

Backend-agnostic physics dataset infrastructure for particle-level generative models.
Supports ROOT and LUND (eager loading) and HDF5 (streaming) backends through a shared base class.

## Package structure

```
data/
├── __init__.py            # Public API exports
├── dataset_base.py        # PhysicsDatasetBase — shared __getitem__, config parsing
├── dataset.py             # EagerDataset (ROOT, LUND) — loads all data at init
├── dataset_h5.py          # StreamingDataset (HDF5) — reads batches on the fly
├── datamodule.py          # ParnassusDataModule — LightningDataModule wrapper
├── sampler.py             # RandomBatchSampler — batch sampler for streaming datasets
├── variable_helper.py     # VariableHelper, CollectionInfo — name parsing
├── computed_vars.py       # ComputedVariables — event sums, charge, impact params, vz_shifted
├── transforms.py          # Sin/cos, one-hot, feature scaling
├── physics_data.py        # Shared assembly: build_ctxt_data, build_fs_data, etc.
└── loaders/
    ├── __init__.py        # open_file_loader — picks the reader from the path
    ├── base.py            # FileLoader ABC
    ├── root_loader.py     # ROOTFileLoader (uproot)
    ├── lund_loader.py     # LUNDFileLoader — generator-level truth from LUND text files
    ├── filtered_loader.py # FilteredFileLoader — exposes only the events a selector keeps
    └── hdf5_loader.py     # HDF5FileLoader (h5py, read_direct)
```

## Architecture

### Core pattern: Template Method

`PhysicsDatasetBase` owns `__getitem__` — all transform, assembly, and output-key
logic lives there. Subclasses only override `_get_collections()` to produce data
in the **common intermediate format**:

```
collection_data:  {collection_name: {bare_var: Tensor(…, N)}}
collection_masks: {collection_name: BoolTensor(…, N)}
event_data:       {var_name: Tensor(…)}
```

Both backends converge to this format, then shared functions handle the rest.

### Data flow

```
YAML config
    │
    ▼
_parse_var_config()  →  ctxt_vars, fs_vars, ctxt_global_vars, ...
    │
    ▼
FileLoader reads raw arrays  →  _get_collections() produces intermediate format
    │
    ▼
__getitem__  calls shared assembly:
    ├── build_ctxt_data()        →  context features  (scaled, sin/cos, one-hot)
    ├── build_fs_data()          →  target features    (scaled)
    ├── build_ctxt_global_data() →  global event vars  (means, ntruth, ht_p, …)
    └── _zero_neutral_vtx()      →  zero vertex vars for neutrals
    │
    ▼
Output dict: {ctxt, ctxt_mask, fs, fs_mask, ctxt_global}
```

### Backend differences

| | EagerDataset (ROOT, LUND) | StreamingDataset (HDF5) |
|---|---|---|
| **Loading** | All data at `__init__` | Per-batch at `__getitem__` |
| **Indexing** | `__getitem__(int)` → single event | `__getitem__(slice)` → batch |
| **Storage** | Ragged arrays + cumsum | Pre-padded 2D arrays |
| **Sorting** | Runtime sort by p + pad | Pre-sorted in HDF5 |
| **Sampler** | Standard PyTorch | `RandomBatchSampler` |
| **Memory** | Full dataset in RAM | Only current batch |

### Variable handling

`VariableHelper` parses prefixed variable names (`truth_p`, `pflow_theta`) to:
- Extract the collection name and bare variable name
- Derive the `FeatureScaler` lookup key (strip prefix, `My` suffix, `npart` normalisation)

`ComputedVariables` computes derived quantities at runtime:
- `charge` — from pdgId + particle class (or read directly from file)
- `vz_shifted` — vertex z shifted by primary vertex z
- Impact parameters (`d0`, `z0`, `d0Sig`, `z0Sig`) — from vertex + track geometry
- Event aggregates from `p`, `theta`, `phi` and `pdgId`: the scalar momentum sum `ht_p`, the
  hadronic (non-electron) momentum sums `Pxh`/`Pyh`, the photon and neutron counts `Ng`/`Nn`,
  and the multiplicities `ntruth`/`npflow`

`EagerDataset` therefore needs `p`, `theta` and `phi` in every loaded collection, and orders
each event's particles by `p`, highest first.

### Transforms (`transforms.py`)

Applied per-variable based on name:
- **phi** → `[sin(phi), cos(phi)]` (when `sin_cos=True`) or standard scaling
- **class** → one-hot (5 categories)
- **charge** → one-hot (3 categories: −1, 0, +1)
- Everything else via `FeatureScaler` with optional transformations.

## Configuration

Two YAML files drive the pipeline:

**Variable config** (`var_dict`) — what to load:
```yaml
event_vars: [eventNumber, Sim_PV_x, ...] # If not defined, loads eventNumber only
vars_to_load:
  truth: [p, theta, phi, pdgId, ...]
  pflow: [p, theta, phi, pdgId, ...]
ctxt_vars: [truth_p, truth_theta, pflow_p, ...]
fs_vars:   [truth_p, truth_theta, ...]
ctxt_global_vars: [means, ntruth, npflow, pflow_ht_p]

# Optional: map file field names → internal names
variable_remapping:
  Sim_PV_x: pv_x
  Sim_PV_y: pv_y
  Sim_PV_z: pv_z
```

**Scale config** (`scale_dict`) — how to scale:
```yaml
p:     {mean: 1.6, std: 1.7}
theta: {mean: 0.4, std: 0.3}
phi:   {mean: 0.0, std: 1.8}
```

## Usage

### EagerDataset (ROOT, LUND)

```python
from parnassus_core.data import EagerDataset

ds = EagerDataset(
    "data.root", var_dict="particle_vars.yaml", scale_dict="var_transform.yaml",
    max_particles=400, train_type="particle", sin_cos=True,
    zero_neutral_vtx=False, reduce_ds=100, mode="train",
)
sample = ds[0]  # dict with ctxt, fs, ctxt_mask, fs_mask, ctxt_global
```

The reader is chosen from the path (`open_file_loader`): `.root` files go to
`ROOTFileLoader`; LUND input (a `.dat`/`.lund` file, a directory of them, or a glob) goes to
`LUNDFileLoader`. LUND files carry generator-level particles only, exposed as the `truth`
collection (`truth_{p,theta,phi,beta,vx,vy,vz,pdgId}`, `ntruth`, and `eventNumber` counting the
events of the input from 0), so they serve `mode="eval"` only. The selection matches the
CLAS12 training samples: final-state particles (`type == 1`), no electron neutrinos, and
`theta >= 1` degree.

An `event_selector` (any object with `branches`, `optional_branches` and a `__call__` returning
a keep mask per chunk; see `loaders.EventSelector`) filters events while reading: a first pass
over the selector's branches picks the events, and only those are loaded. `reduce_ds` then
counts selected events. The evaluation chain uses it for the scatE pre-filter.

### StreamingDataset (HDF5)

```python
from parnassus_core.data import StreamingDataset, RandomBatchSampler

ds = StreamingDataset(
    "data.h5", var_dict="particle_vars.yaml", scale_dict="var_transform.yaml",
    max_particles=400, train_type="particle", sin_cos=True,
    zero_neutral_vtx=False, reduce_ds=100, mode="train",
)
sampler = RandomBatchSampler(ds, batch_size=256, shuffle=True)
for batch_slice in sampler:
    batch = ds[batch_slice]  # same output keys, batched
```

### ParnassusDataModule (Lightning)

```python
from parnassus_core.data import ParnassusDataModule

dm = ParnassusDataModule(
    train_path="train.h5", valid_path="val.h5", test_path="test.h5",
    var_dict="particle_vars.yaml", scale_dict="var_transform.yaml",
    max_particles=400, train_type="particle", sin_cos=True,
    zero_neutral_vtx=False, num_train=100000, num_valid=10000,
    train_batch_size=256, valid_batch_size=512, num_workers=4,
)
# Use with Lightning CLI

def main(args: ArgsType = None) -> None:
    CLI(
        model_class=FlowLightning,
        datamodule_class=ParnassusDataModule,
        save_config_callback=SaveConfig,
        args=args,
        parser_kwargs={
            "default_env": True,
            "fit": {"default_config_files": [f"{config_dir}/base.yaml"]},
        },
    )
```
