import argparse
import glob
from typing import cast

import numpy as np
import awkward as ak

from tqdm import tqdm
from itertools import product

import uproot

from parnassus_core.utils.data_utils import pdgid_class_dict

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


coord_var_list = ["pt", "eta", "phi", "mass"]

vertex_var_list = ["vx", "vy", "vz"]

additional_var_list = [
    "d0",
    "d0Error",
    "z0",
    "z0Error",
    "ip3d",
    "ip3dError",
]

var_dict = {
    "events": ["event"],
    "gens": coord_var_list + ["pdgId"] + vertex_var_list + ["label"],
    "pfcs": coord_var_list
    + ["pdgId"]
    + vertex_var_list
    + additional_var_list
    + ["label"],
}

prefix_dict = {
    "gens": "GenPart_",
    "pfcs": "PFCand_",
}


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


def load_files(filelist, n_events=None, entry_start=0, tr_pt_cut=0, eta_cut=2.7):
    data = {name: {var: None for var in varlist} for name, varlist in var_dict.items()}
    pbar_files = tqdm(filelist, leave=True, position=0)
    for file in pbar_files:
        pbar_files.set_description(f"Loading {file}")
        with uproot.open(file) as f:  # type: ignore
            pbar_trees = tqdm(var_dict.items(), leave=False, position=1)
            for name, varlist in pbar_trees:
                print("1")
                pbar_trees.set_description(f"Loading tree {name}")
                tree = f[name + "/Events"]
                num_events_in_file = tree.num_entries  # type: ignore
                if n_events is None or n_events == -1:
                    n_events = num_events_in_file - entry_start
                elif n_events + entry_start > num_events_in_file:
                    n_events = num_events_in_file - entry_start
                else:
                    n_events = n_events
                # Only load variables that exist in the tree
                available_keys = tree.keys()
                array_names = [
                    prefix_dict.get(name, "") + var
                    for var in varlist
                    if prefix_dict.get(name, "") + var in available_keys
                ]
                arrays = tree.arrays(  # type: ignore
                    array_names,
                    library="np",
                    entry_stop=n_events + entry_start,
                    entry_start=entry_start,
                )
                for var in varlist:
                    array_name = prefix_dict.get(name, "") + var
                    if array_name not in available_keys:
                        continue  # Skip variables not present in file
                    val = arrays[array_name]
                    if data[name][var] is None:
                        data[name][var] = val
                    else:
                        data[name][var] = np.concatenate((data[name][var], val), axis=0)  # type: ignore
    data = cast(dict[str, dict[str, np.ndarray]], data)
    n_events = len(data["events"]["event"])
    data["pfcs"]["class"] = np.copy(data["pfcs"]["pdgId"])
    data["gens"]["class"] = np.copy(data["gens"]["pdgId"])
    n_pfcs = []
    n_gens = []
    for i in tqdm(range(n_events), desc="Processing events"):
        data["pfcs"]["class"][i] = np.vectorize(pdgid_class_dict.get)(
            data["pfcs"]["pdgId"][i]
        ).astype(int)
        data["gens"]["class"][i] = np.vectorize(pdgid_class_dict.get)(
            data["gens"]["pdgId"][i]
        ).astype(int)
        pf_mask = (
            (data["pfcs"]["pt"][i] > 1)
            & (np.abs(data["pfcs"]["eta"][i]) < eta_cut)
            & (data["pfcs"]["class"][i] < 5)
        )
        neutrino_mask = (
            (np.abs(data["gens"]["pdgId"][i]) != 12)
            & (np.abs(data["gens"]["pdgId"][i]) != 14)
            & (np.abs(data["gens"]["pdgId"][i]) != 16)
        )
        tr_mask = (
            (data["gens"]["pt"][i] > tr_pt_cut)
            & (np.abs(data["gens"]["eta"][i]) < eta_cut)
            & neutrino_mask
        )
        for var in (
            coord_var_list
            + ["pdgId", "class"]
            + vertex_var_list
            + additional_var_list
            + ["label"]
        ):
            if var in data["pfcs"] and data["pfcs"][var] is not None:
                data["pfcs"][var][i] = data["pfcs"][var][i][pf_mask]
            if (
                var not in additional_var_list
                and var in data["gens"]
                and data["gens"][var] is not None
            ):
                data["gens"][var][i] = data["gens"][var][i][tr_mask]
        n_pfcs.append(len(data["pfcs"]["pt"][i]))
        n_gens.append(len(data["gens"]["pt"][i]))
    print("Max PFCs: ", max(n_pfcs), "Max Truths: ", max(n_gens))
    print("Min PFCs: ", min(n_pfcs), "Min Truths: ", min(n_gens))
    print("Truth variables: ")
    for var in data["gens"].keys():
        concat_var = np.concatenate(data["gens"][var])
        print(f"Max {var}: ", np.max(concat_var), "Min: ", np.min(concat_var))
    print("PFC variables: ")
    for var in data["pfcs"].keys():
        concat_var = np.concatenate(data["pfcs"][var])
        print(f"Max {var}: ", np.max(concat_var), "Min: ", np.min(concat_var))
    data["pfcs"]["class"] = np.array(data["pfcs"]["class"], dtype=object)
    data["gens"]["class"] = np.array(data["gens"]["class"], dtype=object)

    return data


def var_to_dtype(var):
    if var in ["pdgId", "class", "label"]:
        return np.int32
    else:
        return np.float32


def main():
    args = parser.parse_args()
    if args.input.endswith(".root"):
        input_files = [args.input]
    else:
        input_files = glob.glob(args.input + "/*.root")
    data = load_files(
        input_files,
        args.n_events,
        entry_start=args.entry_start,
        tr_pt_cut=args.pt_cut,
        eta_cut=args.eta_cut,
    )
    print("All variables loaded")

    name_replace = {
        "pflow": "pfcs",
        "truth": "gens",
    }
    out_data_evt = {
        f"{name}_{var}": data[name_replace[name]][var]
        for name, var in product(
            ["pflow", "truth"], coord_var_list + vertex_var_list + ["pdgId", "class"]
        )
        if data[name_replace[name]][var] is not None
    }
    for var in additional_var_list:
        if data["pfcs"][var] is not None:
            out_data_evt[f"pflow_{var}"] = data["pfcs"][var]
    # Add label if present
    if data["pfcs"]["label"] is not None:
        out_data_evt["pflow_label"] = data["pfcs"]["label"]
    if data["gens"]["label"] is not None:
        out_data_evt["truth_label"] = data["gens"]["label"]
    out_data_evt["eventNumber"] = data["events"]["event"]

    if args.n_events == -1:
        n_events = len(data["events"]["event"])
    else:
        n_events = min(args.n_events, len(data["events"]["event"]))

    # Write data in batches to avoid memory issues
    batch_size = 1000
    batch_indices = list(range(0, n_events, batch_size)) + [n_events]

    with uproot.recreate(
        f"{args.output.replace('.root', '')}_{n_events // 1000}k.root"
    ) as f:
        for start, end in tqdm(
            zip(batch_indices[:-1], batch_indices[1:]),
            desc="Writing batches",
            total=len(batch_indices) - 1,
        ):
            pflow_dict = ak.zip(
                {
                    key.split("_")[-1]: ak.values_astype(
                        ak.Array(val[start:end].tolist()),
                        var_to_dtype(key.split("_")[-1]),
                    )
                    for key, val in out_data_evt.items()
                    if key.startswith("pflow")
                }
            )
            truth_dict = ak.zip(
                {
                    key.split("_")[-1]: ak.values_astype(
                        ak.Array(val[start:end].tolist()),
                        var_to_dtype(key.split("_")[-1]),
                    )
                    for key, val in out_data_evt.items()
                    if key.startswith("truth")
                }
            )
            event_number_array = ak.Array(out_data_evt["eventNumber"][start:end])

            if start == 0:
                f["evt_tree"] = {
                    "pflow": pflow_dict,
                    "truth": truth_dict,
                    "eventNumber": event_number_array,
                }
            else:
                f["evt_tree"].extend(
                    {
                        "pflow": pflow_dict,
                        "truth": truth_dict,
                        "eventNumber": event_number_array,
                    }
                )


if __name__ == "__main__":
    main()
