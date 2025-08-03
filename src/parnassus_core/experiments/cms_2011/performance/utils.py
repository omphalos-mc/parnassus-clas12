import uproot
import numpy as np

from tqdm import tqdm


#### Common functions
def load_data(filepath, entry_stop=None, load_cms=False):
    with uproot.open(
        filepath,
        num_workers=8,
    ) as f:
        data = {
            "fastsim": {},
        }
        if load_cms:
            data["pflow"] = {}
            data["truth"] = {}
        tree = f["evt_tree"]
        varnames = tree.keys()
        if not load_cms:
            varnames = [
                var for var in varnames if "pflow" not in var and "truth" not in var
            ]
        arrays = tree.arrays(library="np", entry_stop=entry_stop)
        for var in tree.keys():
            if not load_cms and ("pflow" in var or "truth" in var):
                continue
            arr = arrays[var]
            if "pflow" in var:
                data["pflow"][var.replace("pflow_", "", 1)] = arr
            elif "fastsim" in var:
                data["fastsim"][var.replace("fastsim_", "", 1)] = arr
            elif "truth" in var:
                data["truth"][var.replace("truth_", "", 1)] = arr
            else:
                data[var] = arr
    return data


def zero_neutral_vtx(data):
    for key in ["pflow", "fastsim"]:
        if key not in data:
            continue
        if "vx" not in data[key]:
            continue
        for i in range(len(data[key]["pt"])):
            part_class = data[key]["class"][i]
            data[key]["vx"][i][part_class > 2] = 0
            data[key]["vy"][i][part_class > 2] = 0
            data[key]["vz"][i][part_class > 2] = 0


def get_common_events(data_dict):
    goodEvents = np.array(
        list(
            set.intersection(
                *[set(data["eventNumber"].astype(int)) for data in data_dict.values()]
            )
        ),
        dtype=int,
    )
    return goodEvents


def filter_events(data_dict, n_events=None):
    goodEvents = get_common_events(data_dict)[:n_events]
    for key in data_dict.keys():
        idx_sorted = np.argsort(data_dict[key]["eventNumber"])
        idxs = np.argwhere(
            np.isin(data_dict[key]["eventNumber"][idx_sorted], goodEvents)
        ).flatten()
        for constituent in ["pflow", "fastsim", "truth"]:
            if constituent not in data_dict[key]:
                continue
            for feat_name, value in data_dict[key][constituent].items():
                data_dict[key][constituent][feat_name] = value[idx_sorted][idxs]
        data_dict[key]["eventNumber"] = data_dict[key]["eventNumber"][idx_sorted][idxs]
    print(f"Filtered events: {len(goodEvents)}")


def extract_jet_bkg(data, data_type="jet"):
    new_data = {
        "pflow": {},
        "fastsim": {},
        "truth": {},
    }
    if data_type == "jet":
        mask_fn = lambda x: x > -1  # noqa: E731
    elif data_type == "bkg":
        mask_fn = lambda x: x == -1  # noqa: E731

    for key in ["pflow", "fastsim", "truth"]:
        if key not in data:
            continue
        mask = [mask_fn(x) for x in data[key]["jet_idx"]]
        for subkey, value in data[key].items():
            if "jet_" in subkey and subkey != "jet_idx":
                new_data[key][subkey] = value
                continue
            if subkey in ["npflow", "nfastsim", "ntruth"]:
                new_data[key][subkey] = np.array([np.sum(el) for el in mask])
                continue
            new_arr = []
            for i in range(len(value)):
                new_arr.append(value[i][mask[i]])
            new_data[key][subkey] = np.array(new_arr, dtype=object)
    new_data["eventNumber"] = data["eventNumber"]
    return new_data


def reshape_phi(phi):
    if phi.dtype == np.dtype(object):
        return np.array([reshape_phi(el) for el in phi], dtype=object)
    return np.arctan2(np.sin(phi), np.cos(phi))


def calc_particle_residuals(tr_indices, indices, data, tr_data, varlist=None):
    if varlist is None:
        varlist = ["pt", "eta", "phi", "vx", "vy", "vz"]
    varlist = [var for var in varlist if var in data]

    res = {}
    n_pf = 0
    for i in range(len(indices)):
        if len(indices[i]) == 0 or (len(indices[i]) > len(data["pt"][i])):
            continue
        try:
            n_pf += len(data["pt"][i][indices[i]])
        except Exception as e:
            print(f"Error in calculating n_pf: {e}")
            print(f"{i=}")
            print(f"Indices: {indices[i]}")
            print(f"Data pt: {data['pt'][i]}")
            raise

    for var in varlist:
        res[var] = np.zeros(n_pf)

    curr_pf_len = 0
    for i in tqdm(range(len(indices)), desc="residuals"):
        if len(indices[i]) == 0 or (len(indices[i]) > len(data["pt"][i])):
            continue
        for var in varlist:
            pf_ = data[var][i][indices[i]]
            tr_ = tr_data[var][i][tr_indices[i]]
            if len(pf_) == 0 or len(tr_) == 0:
                continue
            res[var][curr_pf_len : curr_pf_len + len(pf_)] = pf_ - tr_
            if var == "pt":
                res[var][curr_pf_len : curr_pf_len + len(pf_)] /= tr_ + 1e-9
        curr_pf_len += len(pf_)
    for var in varlist:
        res[var] = res[var][:curr_pf_len]
    res["phi"] = reshape_phi(res["phi"])

    return res


def get_classes(tr_indices, indices, data, tr_data):
    """
    Get the classes of the matched particles.

    Parameters:
    -----------
    tr_indices: np.array
        The indices of the truth particles
    indices: np.array
        The indices of the pf particles
    data: dict
        The pf particles
    tr_data: dict
        The truth particles

    Returns:
    --------
    tr_class: np.array
        The classes of the truth particles
    data_class: np.array
        The classes of the pf particles

    """
    tr_class = []
    data_class = []

    for i in range(len(indices)):
        if len(indices[i]) == 0:
            continue
        tr_class.append(tr_data["class"][i][tr_indices[i]])
        data_class.append(data["class"][i][indices[i]])
    data_class = np.concatenate(data_class)
    data_class[data_class == 5] = 3
    return np.concatenate(tr_class), data_class


def calc_jet_residuals(tr_indices, indices, data, tr_data):
    varlist = ["pt", "eta", "phi", "c2", "d2"]
    res = {}
    n_pf = 0
    for i in range(len(indices)):
        if len(indices[i]) == 0:
            continue
        n_pf += len(data["pt"][i][indices[i]])

    for var in varlist:
        res[var] = np.zeros(n_pf)

    curr_pf_len = 0

    for i in tqdm(range(len(indices)), desc="residuals"):
        if len(indices[i]) == 0:
            continue
        for var in varlist:
            pf_ = data[var][i][indices[i]]
            tr_ = tr_data[var][i][tr_indices[i]]
            if len(pf_) == 0 or len(tr_) == 0:
                continue
            if var == "d2":
                res_ = np.log(pf_ + 1e-9) - np.log(tr_ + 1e-9)
            elif var == "phi":
                res_ = reshape_phi(pf_) - reshape_phi(tr_)
            elif var == "pt":
                res_ = (pf_ - tr_) / (tr_ + 1e-9)
            else:
                res_ = pf_ - tr_
            res[var][curr_pf_len : curr_pf_len + len(pf_)] = res_
        curr_pf_len += len(pf_)
    for var in varlist:
        res[var] = res[var][:curr_pf_len]
    res["phi"] = reshape_phi(res["phi"])

    return res


def calc_met_ht(data):
    met_x = []
    met_y = []
    ht = []
    for i in range(len(data["pt"])):
        met_x.append(np.sum(data["pt"][i] * np.cos(data["phi"][i])))
        met_y.append(np.sum(data["pt"][i] * np.sin(data["phi"][i])))
        ht.append(np.sum(data["pt"][i]))
    return np.array(met_x), np.array(met_y), np.array(ht)


#### Loading all files
def load_sample(
    path_dict: dict,
    n_events: int = None,
):
    data_dict = {}
    data_dict_all = {}
    data_dict_jet = {}
    data_dict_bkg = {}

    for key, value in path_dict.items():
        key = key.split("_")[0]  # Clean name
        data_dict[key] = load_data(
            value, entry_stop=n_events, load_cms=True if key == "dl" else False
        )
        zero_neutral_vtx(data_dict[key])
    filter_events(data_dict, n_events=n_events)
    for i, key in enumerate(data_dict.keys()):
        data_dict_jet[key] = extract_jet_bkg(data_dict[key], data_type="jet")
        data_dict_bkg[key] = extract_jet_bkg(data_dict[key], data_type="bkg")
        if "tr" not in data_dict_all and "truth" in data_dict[key]:
            data_dict_all["tr"] = data_dict[key]["truth"]
            data_dict_jet["tr"] = data_dict_jet[key]["truth"]
            data_dict_bkg["tr"] = data_dict_bkg[key]["truth"]

            data_dict_all["pf"] = data_dict[key]["pflow"]
            data_dict_jet["pf"] = data_dict_jet[key]["pflow"]
            data_dict_bkg["pf"] = data_dict_bkg[key]["pflow"]
        data_dict_all[key] = data_dict[key]["fastsim"]
        data_dict_jet[key] = data_dict_jet[key]["fastsim"]
        data_dict_bkg[key] = data_dict_bkg[key]["fastsim"]

    return data_dict_all, data_dict_jet, data_dict_bkg
