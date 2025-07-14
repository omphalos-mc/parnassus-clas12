import argparse
import glob

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
    "gens": coord_var_list + ["pdgId"] + vertex_var_list,
    "pfcs": coord_var_list + ["pdgId"] + vertex_var_list + additional_var_list,
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
        with uproot.open(file) as f:
            pbar_trees = tqdm(var_dict.items(), leave=False, position=1)
            for name, varlist in pbar_trees:
                pbar_trees.set_description(f"Loading tree {name}")
                tree = f[name + "/Events"]
                array_names = [prefix_dict.get(name, "") + var for var in varlist]
                arrays = tree.arrays(
                    array_names,
                    library="np",
                    entry_stop=n_events + entry_start,
                    entry_start=entry_start,
                )
                for var in varlist:
                    val = arrays[prefix_dict.get(name, "") + var]
                    if data[name][var] is None:
                        data[name][var] = val
                    else:
                        data[name][var] = np.concatenate((data[name][var], val), axis=0)
    n_events = len(data["events"]["event"])
    data["pfcs"]["class"] = []
    data["gens"]["class"] = []
    n_gens = []
    n_pfcs = []
    for i in tqdm(range(n_events), desc="Processing events"):
        data["pfcs"]["class"].append(
            np.vectorize(pdgid_class_dict.get)(data["pfcs"]["pdgId"][i]).astype(int)
        )
        data["gens"]["class"].append(
            np.vectorize(pdgid_class_dict.get)(data["gens"]["pdgId"][i]).astype(int)
        )
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
            coord_var_list + ["pdgId", "class"] + vertex_var_list + additional_var_list
        ):
            data["pfcs"][var][i] = data["pfcs"][var][i][pf_mask]
            if var not in additional_var_list:
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


def main():
    args = parser.parse_args()
    if ".root" in args.input:
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
    }
    for var in additional_var_list:
        out_data_evt[f"pflow_{var}"] = data["pfcs"][var]
    out_data_evt["eventNumber"] = data["events"]["event"]

    if args.n_events == -1:
        n_events = len(data["events"]["event"])
    else:
        n_events = min(args.n_events, len(data["events"]["event"]))
    with uproot.recreate(
        f"{args.output.replace('.root', '')}_{n_events // 1000}k.root"
    ) as f:
        f["evt_tree"] = {
            "pflow": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val.tolist())
                    for key, val in out_data_evt.items()
                    if key.startswith("pflow")
                }
            ),
            "truth": ak.zip(
                {
                    key.split("_")[-1]: ak.Array(val.tolist())
                    for key, val in out_data_evt.items()
                    if key.startswith("truth")
                }
            ),
            "eventNumber": out_data_evt["eventNumber"],
        }


if __name__ == "__main__":
    main()
