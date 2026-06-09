# Preprocessing

Tools for converting CMS and Delphes ROOT files, building the Parnassus HDF5
datasets, merging HDF5 files, exporting HepMC, and calculating feature-scaling
statistics.

The main ROOT/HDF5 tools use `awkward`, `h5py`, `hdf5plugin`, `numpy`,
`uproot`, and `tqdm`. The HepMC exporter additionally requires `pyhepmc`.
The legacy PBS helper requires `yaml-include` and access to `qsub`.

## Typical CMS Workflow

The shortest CMS workflow is:

```text
POET CMS ROOT
    -> cms_to_hdf5.py
    -> merge_shuffle_hdf5.py
    -> scaling/calculate_transforms_batched.py
```

The older two-step route remains available:

```text
POET CMS ROOT
    -> cms_to_root.py
    -> root_to_hdf5.py
```

## HDF5 Layout

The HDF5 converters produce event-aligned datasets:

- `events`: structured event-level array
- `truth`: padded generator-particle array
- `pflow`: padded particle-flow array
- `hadrons`: padded heavy-hadron array, when present

All datasets have the same first dimension. Index `i` therefore identifies the
same event in every collection. Particle records contain a `valid` field that
distinguishes real particles from padding. Output uses LZ4 compression through
`hdf5plugin`.

## Direct CMS Conversion

`cms_to_hdf5.py` reads POET trees such as `gens/Events`, `pfcs/Events`,
`events/Events`, and optional hadron or pileup trees, then writes HDF5 directly.

```bash
python -m parnassus_core.preprocessing.cms_to_hdf5 \
  --input /path/to/root/files \
  --output /path/to/output.h5 \
  --n_events -1 \
  --pt_cut 0.25 \
  --pflow-pt-cut 1.0 \
  --eta_cut 2.7 \
  --batch_size 10000 \
  --max_particles '{"truth": 400, "pflow": 400, "hadrons": 50}'
```

Important options:

- `--entry_start`: starting event in each input file
- `--n_events -1`: process all available events
- `--pt_cut`: truth-particle transverse-momentum cut
- `--pflow-pt-cut`: particle-flow transverse-momentum cut
- `--eta_cut`: absolute pseudorapidity limit
- `--max_particles`: JSON mapping of collection names to padded sizes
- `--num_files`: limit the number of ROOT files read from a directory
- `--shuffle`: shuffle independently inside each conversion batch
- `--cut-events`: remove events at or above a particle limit, and events with
  no truth or particle-flow particles

Collections omitted from `--max_particles` retain their defaults: 400 for
`truth` and `pflow`, and 50 for `hadrons`. Quote the JSON argument so the shell
passes it as one value.

## Flat ROOT Conversion

`cms_to_root.py` converts POET CMS files to the older flat `evt_tree` ROOT
layout. It applies the truth, pflow, and eta cuts before writing.

```bash
python -m parnassus_core.preprocessing.cms_to_root \
  --input /path/to/root/files \
  --output /path/to/cms.root \
  --n_events -1 \
  --pt_cut 0.25 \
  --pflow-pt-cut 1.0 \
  --eta_cut 2.7
```

The script appends the processed event count to the output name. For example,
an output base of `cms.root` containing 20,000 events becomes `cms_20k.root`.

`root_to_hdf5.py` converts an `evt_tree` ROOT file, or a directory of such
files, into the standard padded HDF5 layout:

```bash
python -m parnassus_core.preprocessing.root_to_hdf5 \
  --input /path/to/flat/root/files \
  --output /path/to/output.h5 \
  --batch_size 10000 \
  --max_particles '{"truth": 400, "pflow": 400, "hadrons": 50}'
```

It also supports `--nevents`, `--num_files`, `--shuffle`, and `--cut-events`.
When `--output` is omitted, a single input file gets the same basename with an
`.h5` extension; directory input writes `output.h5` inside that directory.

## Merge HDF5 Files

`merge_shuffle_hdf5.py` combines compatible HDF5 files and adds
`source_file_idx` to `events`.

Sequential concatenation finishes one input file before moving to the next:

```bash
python -m parnassus_core.preprocessing.merge_shuffle_hdf5 \
  --input '/path/to/input/*.h5' \
  --output /path/to/merged.h5 \
  --no-shuffle \
  --batch-size 10000 \
  --events-per-file 1400000 \
  --skip-overlap-check
```

With `--events-per-file`, files are named `merged_00000.h5`,
`merged_00001.h5`, and so on. Rotation occurs after a complete batch, so the
target event count is approximate.

Without `--no-shuffle`, batches are sampled proportionally from all inputs and
shuffled. Other options include:

- `--nevents`: limit the total output event count
- `--equal-events`: use the same number of events from every input
- `--seed`: random seed used by shuffled mode
- `--skip-overlap-check`: avoid loading all event numbers during startup

The overlap check can use substantial memory for very large datasets. Skip it
when the inputs are already known to be disjoint.

## Scaling Statistics

`scaling/calculate_transforms_batched.py` calculates event and particle
statistics and derives configured transformations:

```bash
python -m parnassus_core.preprocessing.scaling.calculate_transforms_batched \
  --input /path/to/merged.h5 \
  --config src/parnassus_core/preprocessing/scaling/cms_default_config.yaml \
  --output /path/to/var_transforms.yaml \
  --batch-size 100000
```

Options include `--max-events`, `--skip-particles`, `--split-class`, and
`--recalculate-pv`. See [scaling/README.md](scaling/README.md) for the complete
configuration format and supported transforms.

## Delphes and HepMC

`delphes_to_root.py` converts a Delphes `Delphes` tree to the flat evaluation
`evt_tree` layout:

```bash
python -m parnassus_core.preprocessing.delphes_to_root \
  --input /path/to/delphes.root \
  --num_entries 100000
```

It writes beside the input with a `delphes_` filename prefix and converts
vertex and impact-parameter units from millimeters to centimeters.

`root_to_hepmc.py` exports truth particles from a preprocessed `evt_tree` ROOT
file. It requires the optional `pyhepmc` package:

```bash
python -m parnassus_core.preprocessing.root_to_hepmc \
  --input /path/to/preprocessed.root \
  --num_entries 100000
```

The `.hepmc` output is written beside the input with the same basename.

## WIS PBS Helper

`pbs/submit_cms_to_root.py` is a legacy PBS submission helper for
`cms_to_root.py` intended to run on WIS cluster:

```bash
python src/parnassus_core/preprocessing/pbs/submit_cms_to_root.py \
  --config src/parnassus_core/preprocessing/pbs/train_2012_config.yaml \
  --tag train2012 \
  --dry_run
```

Remove `--dry_run` to submit. The helper writes job scripts and logs under the
configured `log_dir`, records commands in `jobs_<tag>.txt`, and supports
`--rerun` for jobs with empty stdout logs. It targets PBS/qsub systems, not
Slurm.

## File Reference

### Main package

- `__init__.py`: marks `preprocessing` as an importable Python package.
- `cms_to_hdf5.py`: direct POET CMS ROOT to padded HDF5 converter.
- `cms_to_root.py`: POET CMS ROOT to legacy flat `evt_tree` ROOT converter.
- `root_to_hdf5.py`: generic flat `evt_tree` ROOT to padded HDF5 converter.
- `merge_shuffle_hdf5.py`: merge, shuffle, sequentially concatenate, and split
  HDF5 files.
- `delphes_to_root.py`: Delphes ROOT to flat evaluation ROOT converter.
- `root_to_hepmc.py`: truth-particle ROOT to HepMC exporter.

### PBS files

- `pbs/submit_cms_to_root.py`: PBS job generation, submission, and rerun logic.
- `pbs/train_2012_config.yaml`: example input grouping, processing parameters,
  output paths, and PBS resource settings.

### Scaling package

- `scaling/calculate_transforms_batched.py`: user-facing batched statistics
  calculation and YAML writer.
- `scaling/config.py`: validates YAML and defines `RuntimeConfig` and
  `VariableConfig`.
- `scaling/transforms.py`: online statistics and transform implementations.
- `scaling/utils.py`: particle selection, PV recalculation, output formatting,
  and shared scaling helpers.
- `scaling/cms_default_config.yaml`: default CMS variable and transform config.
- `scaling/test.yaml`: example generated scaling-statistics output.
- `scaling/README.md`: detailed scaling configuration documentation.
- `scaling/__init__.py`: marks `scaling` as an importable package.

Use `python -m <module> --help` for the complete CLI of a user-facing script.
