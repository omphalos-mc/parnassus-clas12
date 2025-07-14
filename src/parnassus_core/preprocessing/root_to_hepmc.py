import argparse

import numpy as np
import pyhepmc
import uproot
from tqdm import tqdm


def load_data(filepath, entry_stop=100_000, mode="evt"):
    with uproot.open(
        filepath,
        num_workers=4,
    ) as f:
        data = {}
        tree_name = f"{mode}_tree"
        if tree_name not in f:
            raise ValueError(f"Tree {tree_name} not found in file {filepath}")
        tree = f[tree_name]
        print(f"Loading {mode} data from {filepath}")
        print(f"Entries: {tree.num_entries}")
        for var in tree.keys():
            if "pflow" not in var:
                data[var] = tree[var].array(library="np", entry_stop=entry_stop)
    return data


def get_4momentum(pt, y, phi, mass):
    # e = np.sqrt(pt**2 + mass**2)
    mt = np.sqrt(pt**2 + mass**2)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = mt * np.sinh(y)
    e = mt * np.cosh(y)
    return np.array([px, py, pz, e])


def getParticleHepMC(pt, y, phi, m, pid, status=1):
    p = pyhepmc.GenParticle()
    p.momentum = pyhepmc.FourVector(get_4momentum(pt, y, phi, m))
    p.pid = int(pid)
    p.status = status
    return p


def getVertexHepMC(vx, vy, vz):
    v = pyhepmc.GenVertex()
    v.position = pyhepmc.FourVector([vx, vy, vz, 0])
    return v


def getEventHepMC(event_data):
    event = pyhepmc.GenEvent()
    vtx_dict = {}
    for pt, y, phi, m, pid, vx, vy, vz in zip(*event_data[:-1]):
        particle = getParticleHepMC(pt, y, phi, m, pid)

        vtx = vtx_dict.get(hash((vx, vy, vz)), None)
        if vtx is None:
            vtx = getVertexHepMC(vx, vy, vz)
            particle_in = getParticleHepMC(0.0001, 0, 0, 0, 0)
            event.add_particle(particle_in)
            vtx.add_particle_in(particle_in)
            vtx_dict[hash((vx, vy, vz))] = vtx
        vtx.add_particle_out(particle)
        event.add_particle(particle)
    for vtx in vtx_dict.values():
        event.add_vertex(vtx)
    event.event_number = event_data[-1]
    return event


### doesn't save vertex information!
def getEventHepMC2(event_data):
    event = pyhepmc.GenEvent()
    momentum = get_4momentum(event_data[0], event_data[1], event_data[2], event_data[3])
    parents = np.ones((len(event_data[0]), 2), dtype=int) * -1
    event.from_hepevt(
        event_number=event_data[-1],
        px=momentum[0],
        py=momentum[1],
        pz=momentum[2],
        en=momentum[3],
        m=event_data[3],
        pid=event_data[4],
        status=np.ones_like(event_data[0]),
        parents=parents,
        vx=event_data[5],
        vy=event_data[6],
        vz=event_data[7],
        vt=np.zeros_like(event_data[5]),
    )
    return event


def convert_cms_to_hepmc(data, output_path, prefix, entry_start=0, entry_stop=None):
    hepmc_events = []
    if entry_stop is None:
        entry_stop = len(data["truth_pt"])
    for i in tqdm(range(entry_start, entry_stop)):
        hepmc_events.append(
            getEventHepMC(
                [
                    data[key][i]
                    for key in [
                        "truth_pt",
                        "truth_eta",
                        "truth_phi",
                        "truth_mass",
                        "truth_pdgId",
                        "truth_vx",
                        "truth_vy",
                        "truth_vz",
                        "eventNumber",
                    ]
                ]
            )
        )
    with pyhepmc.open(f"{output_path}/{prefix}.hepmc", "w") as f:
        for event in hepmc_events:
            f.write(event)


parser = argparse.ArgumentParser(
    description="Convert a preprocessed ROOT file to a .hepmc file"
)

parser.add_argument("-i", "--input", type=str, required=True, help="Input ROOT file")
parser.add_argument("-n", "--num_entries", type=int, help="Num entries to convert")

args = parser.parse_args()

data = load_data(args.input, mode=args.mode, entry_stop=args.num_entries)
prefix = args.input.split("/")[-1].split(".")[0]
output_path = args.input.replace(prefix + ".root", "")
total_entries = len(data["truth_pt"])

print("Converting...")
convert_cms_to_hepmc(
    data,
    output_path,
    prefix,
    entry_start=0,
    entry_stop=total_entries,
)
