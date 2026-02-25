import argparse
import glob

import numpy as np
import awkward as ak

from tqdm import tqdm

import uproot

from parnassus_core.utils.data_utils import pdgid_class_dict

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrino

# ============================================================================
# CONFIGURATION
# ============================================================================

# Particle collection configuration (arrays with per-event filtering)
PARTICLE_COLLECTIONS = {
    "gens": {
        "tree": "gens/Events",
        "prefix": "GenPart_",
        "output_name": "truth",
        "variables": ["pt", "eta", "phi", "mass", "pdgId", "vx", "vy", "vz", "label"],
        "optional": False,
    },
    "pfcs": {
        "tree": "pfcs/Events",
        "prefix": "PFCand_",
        "output_name": "pflow",
        "variables": [
            "pt",
            "eta",
            "phi",
            "mass",
            "pdgId",
            "vx",
            "vy",
            "vz",
            "d0",
            "d0Error",
            "z0",
            "z0Error",
            "ip3d",
            "ip3dError",
            "label",
        ],
        "optional": False,
    },
    "hadrons": {
        "tree": "hadrons/Events",
        "prefix": "HeavyHadron_",
        "output_name": "hadrons",
        "variables": ["pt", "eta", "phi", "mass", "pdgId", "vx", "vy", "vz", "label"],
        "optional": True,  # May not be present in all files
    },
}

# Event-level scalar configuration (single value per event, no filtering)
# Format: "input_variable_name": "output_variable_name" (or None to keep same name)
EVENT_SCALARS = {
    "events": {
        "tree": "events/Events",
        "variables": {
            "event": "eventNumber",  # Rename event -> eventNumber
        },
        "optional": False,
    },
    "pu": {
        "tree": "pu/Events",
        "variables": {
            "trueNumPU": None,  # Keep same name
            "actualNumPU": None,
        },
        "optional": True,
    },
    "genvtx": {
        "tree": "genvtx/Events",
        "variables": {
            "Sim_PV_x": None,
            "Sim_PV_y": None,
            "Sim_PV_z": None,
            "Rec_PV_x": None,
            "Rec_PV_y": None,
            "Rec_PV_z": None,
        },
        "optional": True,
    },
}

# ============================================================================
# Example original ROOT file structure
# Tree: events/Events
#   Entries: 9,140
#   Branches (4):
#     run              ->  int32_t
#     luminosityBlock  ->  uint32_t
#     event            ->  uint64_t # event number
#     rhoIso           ->  double
#
# Tree: gens/Events
#   Entries: 9,140
#   Branches (11):
#     numGenPart      ->  int32_t
#     GenPart_pt      ->  std::vector<float>
#     GenPart_eta     ->  std::vector<float>
#     GenPart_mass    ->  std::vector<float>
#     GenPart_pdgId   ->  std::vector<int32_t>
#     GenPart_phi     ->  std::vector<float>
#     GenPart_status  ->  std::vector<int32_t>
#     GenPart_label   ->  std::vector<int32_t>
#     GenPart_vx      ->  std::vector<float>
#     GenPart_vy      ->  std::vector<float>
#     GenPart_vz      ->  std::vector<float>
#
# Tree: hadrons/Events # May not be present
#   Entries: 9,140
#   Branches (11):
#     numHeavyHadron      ->  int32_t
#     HeavyHadron_pt      ->  std::vector<float>
#     HeavyHadron_eta     ->  std::vector<float>
#     HeavyHadron_mass    ->  std::vector<float>
#     HeavyHadron_pdgId   ->  std::vector<int32_t>
#     HeavyHadron_phi     ->  std::vector<float>
#     HeavyHadron_status  ->  std::vector<int32_t>
#     HeavyHadron_label   ->  std::vector<int32_t>
#     HeavyHadron_vx      ->  std::vector<float>
#     HeavyHadron_vy      ->  std::vector<float>
#     HeavyHadron_vz      ->  std::vector<float>
#
# Tree: pfcs/Events
#   Entries: 9,140
#   Branches (16):
#     numPFCand         ->  int32_t
#     PFCand_pt         ->  std::vector<float>
#     PFCand_eta        ->  std::vector<float>
#     PFCand_mass       ->  std::vector<float>
#     PFCand_pdgId      ->  std::vector<int32_t>
#     PFCand_phi        ->  std::vector<float>
#     PFCand_vx         ->  std::vector<float>
#     PFCand_vy         ->  std::vector<float>
#     PFCand_vz         ->  std::vector<float>
#     PFCand_d0         ->  std::vector<float>
#     PFCand_d0Error    ->  std::vector<float>
#     PFCand_z0         ->  std::vector<float>
#     PFCand_z0Error    ->  std::vector<float>
#     PFCand_ip3d       ->  std::vector<float>
#     PFCand_ip3dError  ->  std::vector<float>
#     PFCand_label      ->  std::vector<int32_t>
#
# Tree: pu/Events # May not be present
#   Entries: 9,140
#   Branches (2):
#     trueNumPU      ->  float
#     actualNumPU    ->  float
#
# Tree: genvtx/Events # May not be present
#   Entries: 9,140
#   Branches (6):
#     Sim_PV_x  ->  double
#     Sim_PV_y  ->  double
#     Sim_PV_z  ->  double
#     Rec_PV_x  ->  double
#     Rec_PV_y  ->  double
#     Rec_PV_z  ->  double

# ============================================================================
# ARGUMENT PARSER
# ============================================================================

parser = argparse.ArgumentParser(description="Preprocess CMS files to Parnassus format")
parser.add_argument(
    "-i", "--input", type=str, required=True, help="Input folder or file"
)
parser.add_argument(
    "-o", "--output", type=str, help="Output filename", default="out.root"
)
parser.add_argument("-es", "--entry_start", type=int, help="Starting event", default=0)
parser.add_argument(
    "-n", "--n_events", type=int, help="Number of events to process", default=100
)
parser.add_argument("-pt", "--pt_cut", type=float, help="Truth pt cut", default=0.0)
parser.add_argument("-eta", "--eta_cut", type=float, help="Eta cut", default=2.7)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def var_to_dtype(var):
    """Map variable names to numpy dtypes."""
    if var in ["pdgId", "class", "charge", "label"]:
        return np.int32
    else:
        return np.float32


def load_particle_collections(filelist, n_events=None, entry_start=0):
    """Load particle collection data from ROOT files."""
    data = {
        name: {var: None for var in config["variables"]}
        for name, config in PARTICLE_COLLECTIONS.items()
    }

    pbar_files = tqdm(filelist, leave=True, position=0)
    for file in pbar_files:
        pbar_files.set_description(f"Loading {file}")
        with uproot.open(file) as f:  # type: ignore
            pbar_trees = tqdm(PARTICLE_COLLECTIONS.items(), leave=False, position=1)
            for coll_name, coll_config in pbar_trees:
                pbar_trees.set_description(f"Loading tree {coll_config['tree']}")

                # Check if tree exists
                if coll_config["tree"] not in f:
                    if coll_config["optional"]:
                        print(
                            f"Warning: optional tree {coll_config['tree']} not found in file {file}"
                        )
                        data.pop(coll_name, None)
                        continue
                    else:
                        raise ValueError(
                            f"Required tree {coll_config['tree']} not found in file {file}"
                        )

                tree = f[coll_config["tree"]]
                num_events_in_file = tree.num_entries  # type: ignore

                # Determine actual number of events to load
                if n_events is None or n_events == -1:
                    n_events_to_load = num_events_in_file - entry_start
                elif n_events + entry_start > num_events_in_file:
                    n_events_to_load = num_events_in_file - entry_start
                else:
                    n_events_to_load = n_events

                # Build list of available array names
                available_keys = tree.keys()
                prefix = coll_config["prefix"]
                array_names = [
                    prefix + var
                    for var in coll_config["variables"]
                    if prefix + var in available_keys
                ]

                # Load arrays directly as awkward arrays
                arrays = tree.arrays(  # type: ignore
                    array_names,
                    library="ak",
                    entry_stop=n_events_to_load + entry_start,
                    entry_start=entry_start,
                )

                # Store arrays in data dict
                for var in coll_config["variables"]:
                    array_name = prefix + var
                    if array_name not in available_keys:
                        print(
                            f"Warning: variable {array_name} not found in tree {coll_config['tree']}, file {file}"
                        )
                        continue

                    val = arrays[array_name]
                    if data[coll_name][var] is None:
                        data[coll_name][var] = val
                    else:
                        data[coll_name][var] = ak.concatenate(
                            [data[coll_name][var], val], axis=0
                        )

    return data  # Returns dict with awkward arrays


def load_event_scalars(filelist, n_events=None, entry_start=0):
    """Load event-level scalar data from ROOT files."""
    data = {
        tree_name: {var: None for var in config["variables"].keys()}
        for tree_name, config in EVENT_SCALARS.items()
    }

    for file in tqdm(filelist, desc="Loading event scalars"):
        with uproot.open(file) as f:  # type: ignore
            for tree_name, tree_config in EVENT_SCALARS.items():
                # Check if tree exists
                if tree_config["tree"] not in f:
                    if tree_config["optional"]:
                        print(
                            f"Warning: optional tree {tree_config['tree']} not found in file {file}"
                        )
                        data.pop(tree_name, None)
                        continue
                    else:
                        raise ValueError(
                            f"Required tree {tree_config['tree']} not found in file {file}"
                        )

                tree = f[tree_config["tree"]]
                num_events_in_file = tree.num_entries  # type: ignore

                # Determine actual number of events to load
                if n_events is None or n_events == -1:
                    n_events_to_load = num_events_in_file - entry_start
                elif n_events + entry_start > num_events_in_file:
                    n_events_to_load = num_events_in_file - entry_start
                else:
                    n_events_to_load = n_events

                # Build list of available variables
                available_keys = tree.keys()
                vars_to_load = [
                    var_input
                    for var_input in tree_config["variables"].keys()
                    if var_input in available_keys
                ]

                # Warn about missing variables
                for var_input in tree_config["variables"].keys():
                    if var_input not in available_keys:
                        print(
                            f"Warning: variable {var_input} not found in tree {tree_config['tree']}, file {file}"
                        )

                # Load all variables at once (more efficient), directly as awkward arrays
                if vars_to_load:
                    arrays = tree.arrays(  # type: ignore
                        vars_to_load,
                        library="ak",
                        entry_stop=n_events_to_load + entry_start,
                        entry_start=entry_start,
                    )

                    # Store arrays in data dict
                    for var_input in vars_to_load:
                        val = arrays[var_input]
                        if data[tree_name][var_input] is None:
                            data[tree_name][var_input] = val
                        else:
                            data[tree_name][var_input] = ak.concatenate(
                                [data[tree_name][var_input], val], axis=0
                            )

    return data  # Returns dict with awkward arrays


def process_particle_collections(data, n_events, tr_pt_cut=0.0, eta_cut=2.7):
    """Apply filtering and add derived variables to particle collections."""
    print("Processing particle collections with awkward arrays...")

    # Process PFCs: derive charge/class, then filter
    if "pfcs" in data and data["pfcs"].get("pdgId") is not None:
        pdgId = data["pfcs"]["pdgId"]

        # Derive class (vectorized over all events using ak.unflatten to preserve structure)
        flat_pdgId = ak.flatten(pdgId)
        flat_class = np.vectorize(pdgid_class_dict.get)(ak.to_numpy(flat_pdgId)).astype(
            np.int32
        )
        data["pfcs"]["class"] = ak.unflatten(flat_class, ak.num(pdgId))

        # Derive charge (vectorized over all events)
        data["pfcs"]["charge"] = ak.values_astype(np.sign(pdgId), np.int32)
        # Set charge for neutrals (class > 2) to zero
        neutral_mask = data["pfcs"]["class"] > 2
        data["pfcs"]["charge"] = ak.where(neutral_mask, 0, data["pfcs"]["charge"])

        # Apply filtering (vectorized over all events)
        pf_mask = (
            (data["pfcs"]["pt"] > 1)
            & (np.abs(data["pfcs"]["eta"]) < eta_cut)
            & (data["pfcs"]["class"] < 5)
        )

        # Apply mask to all variables
        for var in list(data["pfcs"].keys()):
            if data["pfcs"][var] is not None:
                data["pfcs"][var] = data["pfcs"][var][pf_mask]

        # Statistics
        n_pfcs = ak.num(data["pfcs"]["pt"])
        print(f"PFCs - Max: {ak.max(n_pfcs)}, Min: {ak.min(n_pfcs)}")

    # Process truth: derive charge/class, then filter
    if "gens" in data and data["gens"].get("pdgId") is not None:
        pdgId = data["gens"]["pdgId"]

        # Derive class (vectorized over all events)
        flat_pdgId = ak.flatten(pdgId)
        flat_class = np.vectorize(pdgid_class_dict.get)(ak.to_numpy(flat_pdgId)).astype(
            np.int32
        )
        data["gens"]["class"] = ak.unflatten(flat_class, ak.num(pdgId))

        # Derive charge (vectorized over all events)
        data["gens"]["charge"] = ak.values_astype(np.sign(pdgId), np.int32)
        # Set charge for neutrals (class > 2) to zero
        neutral_mask = data["gens"]["class"] > 2
        data["gens"]["charge"] = ak.where(neutral_mask, 0, data["gens"]["charge"])

        # Apply filtering: remove neutrinos and apply pt/eta cuts (vectorized)
        neutrino_mask = (
            (np.abs(pdgId) != 12) & (np.abs(pdgId) != 14) & (np.abs(pdgId) != 16)
        )
        tr_mask = (
            (data["gens"]["pt"] > tr_pt_cut)
            & (np.abs(data["gens"]["eta"]) < eta_cut)
            & neutrino_mask
        )

        # Apply mask to all variables
        for var in list(data["gens"].keys()):
            if data["gens"][var] is not None:
                data["gens"][var] = data["gens"][var][tr_mask]

        # Statistics
        n_gens = ak.num(data["gens"]["pt"])
        print(f"Truth - Max: {ak.max(n_gens)}, Min: {ak.min(n_gens)}")

    # Hadrons statistics (no processing needed)
    if "hadrons" in data and data["hadrons"].get("pt") is not None:
        n_hadrons = ak.num(data["hadrons"]["pt"])
        print(f"Hadrons - Max: {ak.max(n_hadrons)}, Min: {ak.min(n_hadrons)}")

    # Print variable ranges
    if "gens" in data:
        print("\nTruth variables:")
        for var in data["gens"].keys():
            if data["gens"][var] is not None:
                flat_var = ak.flatten(data["gens"][var])
                print(
                    f"  {var}: Max={ak.max(flat_var):.4f}, Min={ak.min(flat_var):.4f}"
                )

    if "pfcs" in data:
        print("\nPFC variables:")
        for var in data["pfcs"].keys():
            if data["pfcs"][var] is not None:
                flat_var = ak.flatten(data["pfcs"][var])
                print(
                    f"  {var}: Max={ak.max(flat_var):.4f}, Min={ak.min(flat_var):.4f}"
                )

    return data


def load_files(filelist, n_events=None, entry_start=0, tr_pt_cut=0, eta_cut=2.7):
    """Load and process all data from ROOT files."""
    # Load particle collections
    particle_data = load_particle_collections(filelist, n_events, entry_start)

    # Load event scalars
    scalar_data = load_event_scalars(filelist, n_events, entry_start)

    # Get actual number of events from scalar data (eventNumber is always present)
    for tree_vars in scalar_data.values():
        for var_data in tree_vars.values():
            if var_data is not None:
                n_events_actual = len(var_data)
                break
        else:
            continue
        break
    else:
        raise ValueError("No scalar data found - eventNumber should always be present!")

    # Process particle collections (filtering, charge/class derivation)
    particle_data = process_particle_collections(
        particle_data, n_events_actual, tr_pt_cut, eta_cut
    )

    return particle_data, scalar_data


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    args = parser.parse_args()

    # Get input files
    if args.input.endswith(".root"):
        input_files = [args.input]
    else:
        input_files = glob.glob(args.input + "/*.root")

    print(f"Found {len(input_files)} input file(s)")

    # Load data
    particle_data, scalar_data = load_files(
        input_files,
        args.n_events,
        entry_start=args.entry_start,
        tr_pt_cut=args.pt_cut,
        eta_cut=args.eta_cut,
    )
    print("All data loaded")

    # Determine number of events from scalar data (eventNumber is always present)
    for tree_vars in scalar_data.values():
        for var_data in tree_vars.values():
            if var_data is not None:
                n_events = (
                    len(var_data)
                    if args.n_events == -1
                    else min(args.n_events, len(var_data))
                )
                break
        else:
            continue
        break
    else:
        raise ValueError("No scalar data found - eventNumber should always be present!")

    print(f"Processing {n_events} events")

    # Apply type conversions to awkward arrays (already in awkward format from loading)
    particle_ak_arrays = {}
    scalar_ak_arrays = {}

    # Process particle collections (apply dtypes and zip into records)
    for coll_name, coll_config in PARTICLE_COLLECTIONS.items():
        if coll_name not in particle_data:
            continue

        output_name = coll_config["output_name"]
        coll_dict = {}

        for var in coll_config["variables"] + ["charge", "class"]:
            if (
                var in particle_data[coll_name]
                and particle_data[coll_name][var] is not None
            ):
                # Data already awkward, just apply dtype conversion
                coll_dict[var] = ak.values_astype(
                    particle_data[coll_name][var],
                    var_to_dtype(var),
                )

        if coll_dict:
            particle_ak_arrays[output_name] = ak.zip(coll_dict)

    # Process event scalars (rename and store)
    for tree_name, tree_config in EVENT_SCALARS.items():
        if tree_name not in scalar_data:
            continue

        for var_input, var_output in tree_config["variables"].items():
            if (
                var_input in scalar_data[tree_name]
                and scalar_data[tree_name][var_input] is not None
            ):
                # Use custom output name if specified, otherwise keep input name
                output_var_name = var_output if var_output is not None else var_input
                # Data already awkward, just store it
                scalar_ak_arrays[output_var_name] = scalar_data[tree_name][var_input]

    # Write data in batches to avoid memory issues
    batch_size = 1000
    batch_indices = list(range(0, n_events, batch_size)) + [n_events]

    output_filename = f"{args.output.replace('.root', '')}_{n_events // 1000}k.root"
    print(f"Writing to {output_filename}")

    with uproot.recreate(output_filename) as f:
        for start, end in tqdm(
            zip(batch_indices[:-1], batch_indices[1:]),
            desc="Writing batches",
            total=len(batch_indices) - 1,
        ):
            batch_dict = {}

            # Add particle collections to batch (just slice pre-converted arrays)
            for output_name, ak_array in particle_ak_arrays.items():
                batch_dict[output_name] = ak_array[start:end]

            # Add event scalars to batch (just slice pre-converted arrays)
            for output_var_name, ak_array in scalar_ak_arrays.items():
                batch_dict[output_var_name] = ak_array[start:end]

            # Write or extend tree
            if start == 0:
                f["evt_tree"] = batch_dict
            else:
                f["evt_tree"].extend(batch_dict)  # type: ignore

    print(f"Data successfully written to {output_filename}")


if __name__ == "__main__":
    main()
