import argparse
from typing import Any

import numpy as np
import fastjet as fj
import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray, RecordArray
from awkward.index import Index64

from tqdm import tqdm
from itertools import product

import multiprocessing as mp

import uproot
import sys
import os
from contextlib import contextmanager

from parnassus_core.data.utils.computed_vars import theta_to_eta
from parnassus_core.utils.matching import match_objects, reshape_phi
from parnassus_core.utils.pid import pdgid_to_mass
from parnassus_core.evaluation.postprocess_config import PostProcessConfig

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos

# NOTE: the variable lists (particle/ip/jet/event/matching vars) and all physics
# behavior now live in PostProcessConfig; the defaults there reproduce the values
# that used to be module constants here. See postprocess_config.py.

CATEGORY_MASKS = {
    "all": lambda x: x > -1000,
    "jet": lambda x: x > -1,
    "bkg": lambda x: x == -1,
}

# Derived per-particle four-vector components emitted when cfg.fourvec is set.
# These are computed from pt/eta/phi(/mass), not read from the input files.
FOURVEC_VARS = ["px", "py", "pz", "E"]

parser = argparse.ArgumentParser(description="Postprocess evaluation files")
parser.add_argument("-i", "--input", type=str, required=False, help="Input file")
parser.add_argument(
    "-d", "--data", type=str, required=True, help="Input test data file"
)
parser.add_argument(
    "-o", "--output", type=str, help="Output filename", default="out.root"
)
parser.add_argument("-es", "--entry_start", type=int, help="Starting event", default=0)
parser.add_argument(
    "-n", "--n_events", type=int, help="Number of events to process", default=100
)
parser.add_argument("-dr", type=float, help="Delta R for jet clustering", default=0.5)
parser.add_argument("-df", action="store_true", help="Treat as diffusion file")
parser.add_argument("-dl", action="store_true", help="Treat as Delphes file")
parser.add_argument("--eta", type=float, help="Eta cut", default=3.0)
parser.add_argument("--ip", help="Use impact parameters", action="store_true")
parser.add_argument(
    "-m", "--matching", action="store_true", help="Enable particle matching"
)
parser.add_argument(
    "--dr_cut",
    type=float,
    help="Delta R cut for particle matching",
    default=0.6,
)
parser.add_argument(
    "--join",
    choices=["intersect", "union"],
    default="intersect",
    help="How to combine the -i and -d event sets. 'union' keeps events present in "
    "only one file, writing the missing collection empty for those events.",
)
parser.add_argument(
    "--max_event_number",
    type=int,
    default=None,
    help="Drop joined events above this eventNumber. Use with --join union when the "
    "reference file extends past the last event the eval run consumed.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=None,
    help="Optional PostProcessConfig YAML. When given, it overrides the physics "
    "flags/variable lists; otherwise defaults + the flags above are used.",
)


def init_out_data_dict(cfg: PostProcessConfig):
    collections = ["pflow", "truth", "fastsim"]
    # jet_idx labels each particle by its cluster; only produced when jets are on.
    jet_idx_vars = ["jet_idx"] if cfg.cluster_jets else []
    # px/py/pz/E: derived four-vector components written when cfg.fourvec is set.
    fourvec_vars = FOURVEC_VARS if cfg.fourvec else []
    particle_vars = list(cfg.particle_vars) + fourvec_vars + jet_idx_vars
    out_data_evt = {
        f"{name}_{var}": [] for name, var in product(collections, particle_vars)
    }

    event_vars = list(cfg.event_vars)

    # DIS kinematics are flat per-event scalars, one column per collection/var.
    if cfg.compute_dis:
        event_vars.extend(
            f"{coll}_{v}"
            for coll in ("truth", "pflow", "fastsim")
            for v in cfg.dis_vars
        )

    if cfg.ip:
        particle_vars.extend(cfg.particle_ip_vars)
        for var in cfg.particle_ip_vars:
            out_data_evt[f"pflow_{var}"] = []
            out_data_evt[f"fastsim_{var}"] = []

    if cfg.matching:
        categories = ["all", "jet", "bkg"] if cfg.cluster_jets else ["all"]
        event_vars.extend(
            [
                f"{collection}_hung_cost_{category}"
                for collection, category in product(["fastsim", "pflow"], categories)
            ]
        )
        for category, var in product(categories, cfg.matching_vars):
            out_data_evt[f"fastsim_{var}_{category}"] = []
            out_data_evt[f"pflow_{var}_{category}"] = []
            particle_vars.append(f"{var}_{category}")

    for var in event_vars:
        out_data_evt[var] = []

    if cfg.cluster_jets:
        for name, var in product(collections, cfg.jet_vars):
            out_data_evt[f"{name}_jet_{var}"] = []

    return out_data_evt, particle_vars, event_vars


def load_file(filename, cfg: PostProcessConfig, entry_start=0, n_events=None, df=False):
    with uproot.open(filename, num_workers=4) as file:
        tree = file["evt_tree"]
        tree_keys = set(tree.keys())
        if n_events is None:
            n_events = tree.num_entries
        if n_events == -1:
            n_events = tree.num_entries - entry_start

        if cfg.branch_layout == "auto":
            sep = "." if "truth.phi" in tree_keys else "_"
        else:
            sep = "." if cfg.branch_layout == "nested" else "_"

        pflow_prefix = "fastsim" if df else "pflow"

        def branch(collection, var):
            return f"{collection}{sep}{var}"

        # Read exactly the configured particle_vars, plus the validity mask
        # ("ind") when present (used to strip fastsim padding, not written out).
        read_vars = cfg.particle_vars
        if branch("truth", "ind") in tree_keys:
            read_vars = read_vars + ["ind"]
        ip_vars = list(cfg.particle_ip_vars) if cfg.ip else []

        # Read branch-by-branch: uproot's arrays() expression parser treats the
        # "." in nested sub-branch names ("truth.pt") as attribute access and
        # silently drops them, whereas tree[name].array() resolves both layouts.
        def read(collection, var):
            # Models that generate p instead of pt write no pt branch; their eval
            # output is padded, so pt = p * sin(theta) is a plain array product.
            if var == "pt" and branch(collection, "pt") not in tree_keys:
                return read(collection, "p") * np.sin(read(collection, "theta"))
            return tree[branch(collection, var)].array(
                library="np",
                entry_start=entry_start,
                entry_stop=n_events + entry_start,
            )

        truth_data = {v: read("truth", v) for v in read_vars}
        pflow_data = {v: read(pflow_prefix, v) for v in read_vars + ip_vars}

        event_number = tree["eventNumber"].array(
            library="np",
            entry_start=entry_start,
            entry_stop=n_events + entry_start,
        )
        truth_data["eventNumber"] = event_number
        pflow_data["eventNumber"] = event_number
    return truth_data, pflow_data, event_number


def compute_fourvec(pt, eta, phi, mass=None):
    # Default (mass is None) builds massless 4-vectors: E = pt*cosh(eta). When a
    # per-particle rest mass is supplied (cfg.refine_energy), use the full
    # E = sqrt((pt*cosh(eta))^2 + m^2) so 4-vectors reflect real masses.
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p = pt * np.cosh(eta)
    E = p if mass is None else np.sqrt(p**2 + mass**2)
    return px, py, pz, E


def to_ak(pt, eta, phi, mass=None):
    px, py, pz, E = compute_fourvec(pt, eta, phi, mass=mass)
    return ak.Array(
        {"px": px, "py": py, "pz": pz, "E": E},
        with_name="Momentum4D",
    )


def compute_dis_vars(part_data, cfg: PostProcessConfig):
    """DIS kinematics (Q2, xB, W, y, nu) for one event's collection.

    Reconstructs the event via the electron
    method. The scattered electron is the highest-momentum ``pdgId == 11``
    particle in the collection (fallback: smallest polar angle when momenta are
    not finite). The beam is a fixed electron of energy ``cfg.beam_energy`` along
    +z hitting a proton at rest. Returns a dict keyed by ``cfg.dis_vars``; all
    values are NaN when no electron is present or the value is not requested.
    """
    nan_result = {v: np.nan for v in cfg.dis_vars}
    if "pdgId" not in part_data:
        return nan_result

    pdgId = np.rint(np.asarray(part_data["pdgId"], dtype=float)).astype(int)
    ele = np.flatnonzero(pdgId == 11)
    if len(ele) == 0:
        return nan_result

    pt = np.asarray(part_data["pt"], dtype=float)[ele]
    eta = np.asarray(part_data["eta"], dtype=float)[ele]
    phi = np.asarray(part_data["phi"], dtype=float)[ele]
    p = pt * np.cosh(eta)

    # Prefer the highest-momentum electron; if no momentum is finite, fall back
    # to the smallest polar angle (theta = 2*arctan(exp(-eta))).
    if np.any(np.isfinite(p)):
        sel = int(np.nanargmax(p))
    else:
        theta = 2.0 * np.arctan(np.exp(-eta))
        if not np.any(np.isfinite(theta)):
            return nan_result
        sel = int(np.nanargmin(theta))

    m_e = pdgid_to_mass(11)
    e_pt, e_eta, e_phi = pt[sel], eta[sel], phi[sel]
    e_p = e_pt * np.cosh(e_eta)

    # Scattered electron four-momentum k' = (E', kx, ky, kz).
    kx = e_pt * np.cos(e_phi)
    ky = e_pt * np.sin(e_phi)
    kz = e_pt * np.sinh(e_eta)
    E_prime = np.sqrt(e_p**2 + m_e**2)

    # Beam four-momentum k = (E_beam, 0, 0, p_beam) along +z.
    E_beam = cfg.beam_energy
    p_beam = np.sqrt(max(E_beam**2 - m_e**2, 0.0))

    # q = k - k'; metric (+,-,-,-) so Q2 = -q.q.
    q0 = E_beam - E_prime
    qx = -kx
    qy = -ky
    qz = p_beam - kz
    with np.errstate(divide="ignore", invalid="ignore"):
        Q2 = -(q0**2 - qx**2 - qy**2 - qz**2)
        nu = E_beam - E_prime
        xB = Q2 / (2.0 * cfg.target_mass * nu)
        y = nu / E_beam
        W2 = cfg.target_mass**2 + 2.0 * cfg.target_mass * nu - Q2
        W = np.sqrt(W2) if W2 >= 0 else np.nan

    all_vars = {"Q2": Q2, "xB": xB, "W": W, "y": y, "nu": nu}
    return {v: float(all_vars.get(v, np.nan)) for v in cfg.dis_vars}


def find_repeats(arr):
    return {el: np.argwhere(el == arr).flatten() for el in np.unique(arr)}


def cluster_jets(pt, eta, phi, jetdef, ptmin=20, mass=None):
    from parnassus_core.utils.jet_helper import Jet, get_cluster_sequence

    particles = to_ak(pt, eta, phi, mass=mass)
    cs = get_cluster_sequence(
        jetdef, particles, user_indices=list(range(len(particles)))
    )
    jets = cs.inclusive_jets(ptmin=ptmin)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, 0.5, calc_substructure=True) for j in jets]
    jets = [j for j in jets if j.nconstituents >= 2]

    used_indices = set()

    jet_idxs = np.zeros(len(pt), dtype=int)
    for jet_idx, jet in enumerate(jets):
        particle_idx = jet.constituents_idx
        jet_idxs[particle_idx] = jet_idx
        used_indices.update(particle_idx)
    particle_idx = np.arange(len(pt))
    particle_idx = particle_idx[~np.isin(particle_idx, list(used_indices))]
    jet_idxs[particle_idx] = -1

    return jets, jet_idxs


def match_event(
    truth_data,
    fs_evt_data,
    pflow_data,
    truth_jet_idx,
    pflow_jet_idx,
    fastsim_jet_idx,
    dr_cut=0.6,
    categories=None,
):
    # With jets off only the "all" category is meaningful (jet/bkg key off jet_idx).
    if categories is None:
        categories = list(CATEGORY_MASKS.keys())
    out_dict = dict()
    tr_eta = truth_data["eta"]
    tr_phi = reshape_phi(truth_data["phi"])

    fs_eta = fs_evt_data["eta"]
    fs_phi = reshape_phi(fs_evt_data["phi"])
    fs_len = len(fs_eta)

    pf_eta = pflow_data["eta"]
    pf_phi = reshape_phi(pflow_data["phi"])
    pf_len = len(pf_eta)
    for category in categories:
        mask = CATEGORY_MASKS[category]
        tr_mask = mask(truth_jet_idx)
        fs_mask = mask(fastsim_jet_idx)
        pf_mask = mask(pflow_jet_idx)
        fs_input_idx, fs_truth_idx, fs_hung_cost = match_objects(
            fs_eta[fs_mask],
            fs_phi[fs_mask],
            tr_eta[tr_mask],
            tr_phi[tr_mask],
            dr_cut=dr_cut,
            pad_to_len=fs_len,
        )
        pf_input_idx, pf_truth_idx, pf_hung_cost = match_objects(
            pf_eta[pf_mask],
            pf_phi[pf_mask],
            tr_eta[tr_mask],
            tr_phi[tr_mask],
            dr_cut=dr_cut,
            pad_to_len=pf_len,
        )
        out_dict[f"fastsim_hung_cost_{category}"] = fs_hung_cost
        out_dict[f"fastsim_tr_idx_{category}"] = fs_truth_idx
        out_dict[f"fastsim_pf_idx_{category}"] = fs_input_idx

        out_dict[f"pflow_hung_cost_{category}"] = pf_hung_cost
        out_dict[f"pflow_tr_idx_{category}"] = pf_truth_idx
        out_dict[f"pflow_pf_idx_{category}"] = pf_input_idx
    return out_dict


def process_events(
    batch_event_numbers,
    evt_truth_data,
    evt_pflow_data,
    fs_data,
    fs_truth_data,
    has_ref,
    has_fs,
    cfg: PostProcessConfig,
):
    dl_flag = cfg.dl
    eta_cut = cfg.eta_cut
    if cfg.cluster_jets:
        jetdef = fj.JetDefinition(fj.antikt_algorithm, cfg.jet_radius)
    categories = ["all", "jet", "bkg"] if cfg.cluster_jets else ["all"]

    out_data_evt, _, _ = init_out_data_dict(cfg)

    for i in range(len(batch_event_numbers)):
        # "ind" (the validity mask) is used to strip fastsim padding below but is
        # never written out, so it is excluded from the per-event collections.
        if has_ref[i] or not fs_truth_data:
            truth_data = {
                var: evt_truth_data[var][i]
                for var in evt_truth_data.keys()
                if var not in ("eventNumber", "ind")
            }
        else:
            # Union join: the reference file does not cover this event, so take
            # truth from the eval file instead. That collection is padded, so its
            # own "ind" mask has to be applied here (the reference file is jagged
            # and needs none).
            tr_ind = fs_truth_data["ind"][i].astype(bool)
            truth_data = {
                var: fs_truth_data[var][i][tr_ind]
                for var in fs_truth_data.keys()
                if var not in ("eventNumber", "ind")
            }
        pflow_data = {
            var: evt_pflow_data[var][i]
            for var in evt_pflow_data.keys()
            if var not in ("eventNumber", "ind")
        }
        truth_event_number = evt_truth_data["eventNumber"][i]
        pflow_event_number = evt_pflow_data["eventNumber"][i]
        if truth_event_number != pflow_event_number:
            raise ValueError("Event numbers do not match")

        if not dl_flag:
            fs_ind = fs_data["ind"][i].astype(bool)
            fs_evt_data = {
                var: fs_data[var][i][fs_ind]
                for var in fs_data.keys()
                if var not in ("eventNumber", "ind")
            }
        else:
            fs_evt_data = {
                var: fs_data[var][i]
                for var in fs_data.keys()
                if var not in ("eventNumber", "ind")
            }

        # 'eta' (and 'pt', when particle_vars has none) is derived scaffolding for
        # the geometry helpers below (acceptance, four-vectors, matching, jets),
        # which are written in terms of pseudorapidity and pt. 'theta' and 'p'
        # remain the variables of record.
        for part_data in (truth_data, pflow_data, fs_evt_data):
            if "theta" in part_data and "eta" not in part_data:
                part_data["eta"] = theta_to_eta(part_data["theta"])
            if "pt" not in part_data and {"p", "theta"} <= part_data.keys():
                part_data["pt"] = part_data["p"] * np.sin(part_data["theta"])

        fs_evt_mask = (
            (fs_evt_data["pt"] > cfg.pt_min)
            & (fs_evt_data["pt"] < cfg.pt_max)
            & (np.abs(fs_evt_data["eta"]) <= eta_cut)
        )
        for key, val in fs_evt_data.items():
            fs_evt_data[key] = val[fs_evt_mask]
        pf_evt_mask = np.abs(pflow_data["eta"]) <= eta_cut
        for key, val in pflow_data.items():
            pflow_data[key] = val[pf_evt_mask]

        # Per-particle rest masses from pdgId (cfg.refine_energy); pdgId is already
        # filtered by the same acceptance masks as pt/eta/phi above, so the masses
        # stay aligned. None → massless 4-vectors (default).
        if cfg.refine_energy:
            # if no pdgId present, assume pion mass for a better approximation
            if "pdgId" not in truth_data:
                tr_mass = np.full(len(truth_data["pt"]), 0.13957)
            else:
                tr_mass = pdgid_to_mass(truth_data["pdgId"])

            if "pdgId" not in pflow_data:
                pf_mass = np.full(len(pflow_data["pt"]), 0.13957)
            else:
                pf_mass = pdgid_to_mass(pflow_data["pdgId"])

            if "pdgId" not in fs_evt_data:
                fs_mass = np.full(len(fs_evt_data["pt"]), 0.13957)
            else:
                fs_mass = pdgid_to_mass(fs_evt_data["pdgId"])
        else:
            tr_mass = pf_mass = fs_mass = None

        # Jet clustering
        if cfg.cluster_jets:
            pf_jets, pf_jet_indices = cluster_jets(
                pflow_data["pt"],
                pflow_data["eta"],
                pflow_data["phi"],
                jetdef,
                mass=pf_mass,
            )
            tr_jets, tr_jet_indices = cluster_jets(
                truth_data["pt"],
                truth_data["eta"],
                truth_data["phi"],
                jetdef,
                mass=tr_mass,
            )
            fs_jets, fs_jet_indices = cluster_jets(
                fs_evt_data["pt"],
                fs_evt_data["eta"],
                fs_evt_data["phi"],
                jetdef,
                mass=fs_mass,
            )
        else:
            # No jets: label every particle as unclustered so the "all" category
            # (jet_idx > -1000) still selects everything during matching.
            tr_jet_indices = -np.ones(len(truth_data["pt"]), dtype=int)
            pf_jet_indices = -np.ones(len(pflow_data["pt"]), dtype=int)
            fs_jet_indices = -np.ones(len(fs_evt_data["pt"]), dtype=int)

        # The derived 'eta' above is internal only; theta is the variable of record.
        # Empty for eta-native datasets (CMS), so those still write 'eta'.
        derived = {"eta"} if "theta" in truth_data else set()
        if "pt" not in cfg.particle_vars:
            derived.add("pt")
        for key in truth_data.keys():
            if key in derived:
                continue
            out_data_evt[f"truth_{key}"].append(truth_data[key])
        for key in pflow_data.keys():
            if key in derived:
                continue
            out_data_evt[f"pflow_{key}"].append(pflow_data[key])
            out_data_evt[f"fastsim_{key}"].append(fs_evt_data[key])

        # Derived px/py/pz/E per collection (written independently of jets). Uses
        # the same masses as jet clustering, so E matches refine_energy.
        if cfg.fourvec:
            for name, data, mass in (
                ("truth", truth_data, tr_mass),
                ("pflow", pflow_data, pf_mass),
                ("fastsim", fs_evt_data, fs_mass),
            ):
                px, py, pz, E = compute_fourvec(
                    data["pt"], data["eta"], data["phi"], mass=mass
                )
                out_data_evt[f"{name}_px"].append(px)
                out_data_evt[f"{name}_py"].append(py)
                out_data_evt[f"{name}_pz"].append(pz)
                out_data_evt[f"{name}_E"].append(E)

        if cfg.cluster_jets:
            out_data_evt["truth_jet_idx"].append(tr_jet_indices)
            out_data_evt["pflow_jet_idx"].append(pf_jet_indices)
            out_data_evt["fastsim_jet_idx"].append(fs_jet_indices)

            for name, jet_collection in zip(
                ["pflow", "truth", "fastsim"], [pf_jets, tr_jets, fs_jets]
            ):
                out_data_evt[f"{name}_jet_pt"].append(
                    np.array([jet.pt() for jet in jet_collection])
                )
                out_data_evt[f"{name}_jet_eta"].append(
                    np.array([jet.eta() for jet in jet_collection])
                )
                out_data_evt[f"{name}_jet_phi"].append(
                    np.array([jet.phi() for jet in jet_collection])
                )
                out_data_evt[f"{name}_jet_d2"].append(
                    np.array([jet.substructure["d2"] for jet in jet_collection])
                )
                out_data_evt[f"{name}_jet_c2"].append(
                    np.array([jet.substructure["c2"] for jet in jet_collection])
                )

        if cfg.matching:
            matched_data = match_event(
                truth_data,
                fs_evt_data,
                pflow_data,
                tr_jet_indices,
                pf_jet_indices,
                fs_jet_indices,
                dr_cut=cfg.dr_cut,
                categories=categories,
            )
            for key, val in matched_data.items():
                out_data_evt[key].append(val)

        # DIS kinematics per collection (flat per-event scalars). pdgId-derived
        # masses (cfg.refine_energy) are reused inside compute_dis_vars via
        # pdgid_to_mass; here we just pass the acceptance-filtered collections.
        if cfg.compute_dis:
            for name, data in (
                ("truth", truth_data),
                ("pflow", pflow_data),
                ("fastsim", fs_evt_data),
            ):
                dis = compute_dis_vars(data, cfg)
                for v in cfg.dis_vars:
                    out_data_evt[f"{name}_{v}"].append(dis[v])

        out_data_evt["eventNumber"].append(truth_event_number)
    return out_data_evt


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def process_events_wrapper(args):
    with stdout_redirected():
        return process_events(*args)


def _empty_like_elements(val):
    """Empty per-event array matching the dtype of a jagged column's elements.

    Taken from the first non-empty element so that, e.g., an int32 pdgId column
    is not promoted to float when the empty fills are concatenated at write time.
    """
    for el in val:
        el = np.asarray(el)
        if el.size:
            return np.array([], dtype=el.dtype)
    return np.array([], dtype=np.float64)


def _align_to_batch(batch_event_numbers, source_event_numbers, data):
    """Reindex a source's columns onto ``batch_event_numbers``.

    Returns ``(aligned, present)``. Row ``i`` of every returned column is the event
    ``batch_event_numbers[i]``; events the source does not contain get an empty
    array and ``present[i] is False``.

    Replaces a plain ``np.isin`` mask, which keeps each source in *its own* order
    and so only lines the sources up when the files happen to be sorted the same
    way. Reindexing makes the correspondence explicit, and is what allows the
    "union" join to mix sources that cover different event sets.
    """
    source_event_numbers = np.asarray(source_event_numbers)
    if len(source_event_numbers) == 0:
        present = np.zeros(len(batch_event_numbers), dtype=bool)
        idx = np.zeros(len(batch_event_numbers), dtype=int)
    else:
        idx = np.clip(
            np.searchsorted(source_event_numbers, batch_event_numbers),
            0,
            len(source_event_numbers) - 1,
        )
        present = source_event_numbers[idx] == batch_event_numbers

    aligned = {}
    missing = np.flatnonzero(~present)
    for key, val in data.items():
        if key == "eventNumber":
            aligned[key] = np.asarray(batch_event_numbers)
            continue
        picked = val[idx].copy()
        if missing.size:
            if picked.ndim > 1:
                # Padded rectangular source (the eval file, padded to
                # max_particles). "No particles" is expressed by zeroing the row:
                # the validity mask "ind" goes to 0, and the ind-masking in
                # process_events then yields an empty event.
                picked[missing] = 0
            else:
                # Jagged source (the reference file): drop in a zero-length array.
                empty = _empty_like_elements(val)
                for j in missing:
                    picked[j] = empty
        aligned[key] = picked
    return aligned, present


def extract_batch_data(
    batch_event_numbers,
    evt_event_numbers,
    fs_event_numbers,
    evt_truth_data,
    evt_pflow_data,
    fs_data,
    fs_truth_data=None,
):
    # Aligned in both join modes: under "intersect" every event is present in both
    # files so `present` is all-True and the result matches the previous
    # mask-based slicing exactly.
    evt_truth_data, has_ref = _align_to_batch(
        batch_event_numbers, evt_event_numbers, evt_truth_data
    )
    evt_pflow_data, _ = _align_to_batch(
        batch_event_numbers, evt_event_numbers, evt_pflow_data
    )
    fs_data, has_fs = _align_to_batch(batch_event_numbers, fs_event_numbers, fs_data)
    # Fallback truth for events the reference file does not cover (union join):
    # the eval file carries its own truth collection for exactly those events.
    fs_truth_data = (
        {}
        if fs_truth_data is None
        else _align_to_batch(batch_event_numbers, fs_event_numbers, fs_truth_data)[0]
    )
    return evt_truth_data, evt_pflow_data, fs_data, fs_truth_data, has_ref, has_fs


def _make_collection_array(var_data: dict[str, list[Any]]) -> ak.Array:
    """Build a variable-length record array with shared offsets."""
    if not var_data:
        return ak.Array([])

    first = next(iter(var_data.values()))
    counts = np.fromiter((len(arr) for arr in first), dtype=np.int64, count=len(first))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    fields = list(var_data.keys())
    record = RecordArray(
        [NumpyArray(np.concatenate(var_data[field])) for field in fields],  # type: ignore[arg-type]
        fields,
    )
    return ak.Array(ListOffsetArray(Index64(offsets), record))


def main():
    args = parser.parse_args()

    cfg = (
        PostProcessConfig.from_yaml(args.config)
        if args.config
        else PostProcessConfig.from_cli(args)
    )

    print(f"Loading event file: {args.input.split('/')[-1]}")
    # The eval file's own truth collection is only needed by the "union" join, as
    # the fallback for events the reference file does not cover.
    fs_truth_data, fs_data, fs_event_number = load_file(
        args.input, cfg, n_events=args.n_events, df=cfg.df
    )
    print(f"Loading data file: {'/'.join(args.data.split('/')[-2:])}")
    evt_truth_data, evt_pflow_data, evt_event_number = load_file(
        args.data, cfg, n_events=args.n_events
    )

    print("All variables loaded")

    out_data_evt, particle_vars, event_vars = init_out_data_dict(cfg)

    if cfg.join == "union":
        # Events the eval file does not cover are zeroed out and rely on the "ind"
        # mask to come back empty; -dl skips that mask, so the two cannot combine.
        if cfg.dl:
            print("ERROR: join='union' is incompatible with -dl (no validity mask to")
            print(" mark padded rows empty). Exiting...")
            sys.exit(1)
        goodEventNumbers = np.union1d(evt_event_number, fs_event_number)
    else:
        goodEventNumbers = np.intersect1d(evt_event_number, fs_event_number)
    # Default the union cap to the eval file's last event. The reference file is read
    # over its own -ne window and routinely extends past what the chain was actually
    # offered, and those extra reference-only rows have no fastsim counterpart by
    # construction. An explicit cfg.max_event_number still wins, and still applies to
    # both joins.
    max_event_number = cfg.max_event_number
    derived_cap = False
    if max_event_number is None and cfg.join == "union" and len(fs_event_number):
        max_event_number = int(np.max(fs_event_number))
        derived_cap = True
    if max_event_number is not None:
        goodEventNumbers = goodEventNumbers[goodEventNumbers <= max_event_number]

    if args.n_events > len(goodEventNumbers) or args.n_events == -1:
        args.n_events = len(goodEventNumbers)
    if len(goodEventNumbers) == 0:
        print(
            "ERROR: Event numbers do not match in the input files,"
            " are you sure you are using the same data file as for evaluation?\n"
            " Exiting..."
        )
        sys.exit(1)
    if cfg.join == "union":
        n_both = np.isin(goodEventNumbers, evt_event_number) & np.isin(
            goodEventNumbers, fs_event_number
        )
        n_fs_only = np.isin(goodEventNumbers, fs_event_number) & ~n_both
        n_ref_only = np.isin(goodEventNumbers, evt_event_number) & ~n_both
        print(
            f"Union join: {len(goodEventNumbers)} events"
            f" = {int(n_both.sum())} in both"
            f" + {int(n_fs_only.sum())} fastsim-only (empty pflow)"
            f" + {int(n_ref_only.sum())} reference-only (empty fastsim)"
        )
        if max_event_number is not None:
            source = "derived from -i" if derived_cap else "from config"
            print(f"  (capped at eventNumber <= {max_event_number}, {source})")
    print(
        f"Having {len(goodEventNumbers)} good events, processing {args.n_events} of them"
    )

    batch_size = 1000
    n_batches = args.n_events // batch_size
    n_batches += 1 if args.n_events % batch_size != 0 else 0

    input_batches = np.array_split(goodEventNumbers[: args.n_events], n_batches)
    input_batched_data = [
        (
            batch,
            *extract_batch_data(
                batch,
                evt_event_number,
                fs_event_number,
                evt_truth_data,
                evt_pflow_data,
                fs_data,
                fs_truth_data if cfg.join == "union" else None,
            ),
            cfg,
        )
        for batch in input_batches
    ]
    with mp.Pool(processes=20) as pool:
        results = list(
            tqdm(
                pool.imap(process_events_wrapper, input_batched_data),
                total=n_batches,
            )
        )

    for result in results:
        for key, val in result.items():
            out_data_evt[key].extend(val)

    output_path = f"{args.output.replace('.root', '')}_{args.n_events // 1000}k_test_eta{int(cfg.eta_cut * 10)}Eval.root"
    with uproot.recreate(output_path) as f:
        write_batch_size = 1000
        total_events = len(out_data_evt[next(iter(out_data_evt.keys()))])

        particle_collections = ["truth", "pflow", "fastsim"]
        particle_field_map = {
            collection: [
                (key.replace(f"{collection}_", "", 1), key)
                for key in out_data_evt
                if key.startswith(collection)
                and key.replace(f"{collection}_", "", 1) in particle_vars
            ]
            for collection in particle_collections
        }
        jet_field_map = (
            {
                collection: [
                    (key.replace(f"{collection}_", "", 1), key)
                    for key in out_data_evt
                    if key.startswith(f"{collection}_jet") and "idx" not in key
                ]
                for collection in particle_collections
            }
            if cfg.cluster_jets
            else {}
        )

        for start_idx in tqdm(
            range(0, total_events, write_batch_size), desc="Writing to file"
        ):
            end_idx = min(start_idx + write_batch_size, total_events)

            particle_data = {
                collection: _make_collection_array(
                    {
                        field: out_data_evt[key][start_idx:end_idx]
                        for field, key in particle_field_map[collection]
                    }
                )
                for collection in particle_collections
            }
            jet_data = {
                f"{collection}_jets": _make_collection_array(
                    {
                        field: out_data_evt[key][start_idx:end_idx]
                        for field, key in jet_field_map[collection]
                    }
                )
                for collection in jet_field_map
            }
            event_data = {
                key: np.asarray(val[start_idx:end_idx])
                for key, val in out_data_evt.items()
                if key in event_vars
            }

            batch_data = event_data | particle_data | jet_data

            if start_idx == 0:
                f["evt_tree"] = batch_data
            else:
                f["evt_tree"].extend(batch_data)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
