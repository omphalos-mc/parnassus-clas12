import argparse

import numpy as np
import uproot
import awkward as ak

from parnassus_core.utils.data_utils import pdgid_class_dict

var_map = {
    # Truth variables
    "truth_pt": "Particle/Particle.PT",
    "truth_eta": "Particle/Particle.Rapidity",
    "truth_phi": "Particle/Particle.Phi",
    "truth_pid": "Particle/Particle.PID",
    "truth_vx": "Particle/Particle.X",
    "truth_vy": "Particle/Particle.Y",
    "truth_vz": "Particle/Particle.Z",
    # Particle flow variables
    "pflow_pt": "ParticleFlowCandidate/ParticleFlowCandidate.PT",
    "pflow_eta": "ParticleFlowCandidate/ParticleFlowCandidate.Eta",
    "pflow_phi": "ParticleFlowCandidate/ParticleFlowCandidate.Phi",
    "pflow_pid": "ParticleFlowCandidate/ParticleFlowCandidate.PID",
    "pflow_vx": "ParticleFlowCandidate/ParticleFlowCandidate.X",
    "pflow_vy": "ParticleFlowCandidate/ParticleFlowCandidate.Y",
    "pflow_vz": "ParticleFlowCandidate/ParticleFlowCandidate.Z",
    # Impact parameters
    "pflow_d0": "ParticleFlowCandidate/ParticleFlowCandidate.D0",
    "pflow_d0Error": "ParticleFlowCandidate/ParticleFlowCandidate.ErrorD0",
    "pflow_z0": "ParticleFlowCandidate/ParticleFlowCandidate.DZ",
    "pflow_z0Error": "ParticleFlowCandidate/ParticleFlowCandidate.ErrorDZ",
}


def load_data(filepath, entry_stop=None):
    with uproot.open(
        filepath,
        num_workers=8,
    ) as f:
        truth_data = {}
        pflow_data = {}
        tree = f["Delphes"]
        for key, val in var_map.items():
            if val not in tree:
                raise ValueError(f"Variable {val} not found in file {filepath}")
            arr = tree[val].array(library="np", entry_stop=entry_stop)
            if "pid" in key:
                arr = np.array(
                    [
                        np.vectorize(pdgid_class_dict.get)(pid).astype(int)
                        for pid in arr
                    ],
                    dtype=object,
                )
            key = key.replace("pid", "class")

            if "truth" in key:
                truth_data[key.replace("truth_", "")] = arr
            else:
                pflow_data[key.replace("pflow_", "")] = arr
        event_number = np.concatenate(
            tree["Event/Event.Number"].array(library="np", entry_stop=entry_stop)
        )
    return truth_data, pflow_data, event_number


parser = argparse.ArgumentParser(description="Convert a Delphes file to a eval file")

parser.add_argument("-i", "--input", type=str, required=True, help="Input Delphes file")
parser.add_argument("-n", "--num_entries", type=int, help="Num entries to convert")

args = parser.parse_args()

print(f"Loading data from {args.input}...")
truth_data, pflow_data, event_number = load_data(
    args.input, entry_stop=args.num_entries
)
prefix = args.input.split("/")[-1].split(".")[0]
output_path = args.input.replace(prefix, "delphes_" + prefix)
total_entries = len(truth_data["pt"])

print(f"Converting {total_entries}...")
with uproot.recreate(output_path) as f:
    f["evt_tree"] = {
        "truth": ak.zip(
            {key: ak.Array(truth_data[key].tolist()) for key in truth_data.keys()}
        ),
        "pflow": ak.zip(
            {key: ak.Array(pflow_data[key].tolist()) for key in pflow_data.keys()}
        ),
        "eventNumber": ak.Array(event_number),
    }
