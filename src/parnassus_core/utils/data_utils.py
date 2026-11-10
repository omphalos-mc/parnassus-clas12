from dataclasses import dataclass

import numpy as np
import uproot
import vector as vec
import tqdm

from parnassus_core.utils.jet_helper import get_pt_sorted_jets
from parnassus_core.utils.pid import pdgid_class_dict


def normalize_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def convert_flow(
    pf_pt,
    pf_eta,
    pf_phi,
    pf_ind,
    fs_pt,
    fs_eta,
    fs_phi,
    fs_ind,
    tr_pt,
    tr_eta,
    tr_phi,
    tr_ind,
):
    pf_pt_comp = []
    pf_eta_comp = []
    pf_phi_comp = []
    fs_pt_comp = []
    fs_eta_comp = []
    fs_phi_comp = []
    tr_pt_comp = []
    tr_eta_comp = []
    tr_phi_comp = []

    for i in range(len(pf_pt)):
        fs_mask = (
            (fs_ind[i] > 0.5)
            & (fs_pt[i] > 1000)
            & (fs_pt[i] < 3e6)
            & (np.abs(fs_eta[i]) < 3)
        )
        pf_mask = pf_ind[i] > 0.5
        tr_mask = tr_ind[i] > 0.5

        pf_pt_comp.append(pf_pt[i][pf_mask])
        pf_eta_comp.append(pf_eta[i][pf_mask])
        pf_phi_comp.append(pf_phi[i][pf_mask])

        fs_pt_comp.append(fs_pt[i][fs_mask])
        fs_eta_comp.append(fs_eta[i][fs_mask])
        fs_phi_comp.append(fs_phi[i][fs_mask])

        tr_pt_comp.append(tr_pt[i][tr_mask])
        tr_eta_comp.append(tr_eta[i][tr_mask])
        tr_phi_comp.append(tr_phi[i][tr_mask])

    return (
        np.array(pf_pt_comp, dtype=object),
        np.array(pf_eta_comp, dtype=object),
        np.array(pf_phi_comp, dtype=object),
        np.array(fs_pt_comp, dtype=object),
        np.array(fs_eta_comp, dtype=object),
        np.array(fs_phi_comp, dtype=object),
        np.array(tr_pt_comp, dtype=object),
        np.array(tr_eta_comp, dtype=object),
        np.array(tr_phi_comp, dtype=object),
    )


@dataclass
class GenParticle:
    "Class storing information about generic particle"

    # Particle properties
    pt: float = None
    eta: float = None
    phi: float = None
    mass: float = None
    pdgId: float = None
    classId: int = None
    charge: int = None
    status: int = None
    particle_jet_idx: int = None

    def __repr__(self) -> str:
        return (
            f"Particle with $p_T$={self.pt}, $\eta$={self.eta}, $\phi$={self.phi}, "
            f"mass={self.mass}, pdgId={self.pdgId}, classId={self.classId}, "
            f"charge={self.charge}, status={self.status}, jet_idx={self.particle_jet_idx}"
        )


@dataclass
class GenParticleCollection:
    """Class storing information about a collection of generic particles.

    This class represents a collection of generic particles and provides methods to access and manipulate their properties.

    Attributes:
        name (str): The name of the particle collection.
        pt (np.ndarray): The transverse momentum of the particles.
        eta (np.ndarray): The pseudorapidity of the particles.
        phi (np.ndarray): The azimuthal angle of the particles.
        mass (np.ndarray): The mass of the particles.
        pdgId (np.ndarray): The PDG ID of the particles.
        classId (np.ndarray): The class ID of the particles.
        n (int): The number of particles in the collection.
        charge (np.ndarray): The charge of the particles.

    Methods:
        __post_init__: Initializes additional properties of the particle collection.
        __len__: Returns the number of particles in the collection.
        __repr__: Returns a string representation of the particle collection.
        __getitem__: Returns the properties of a particle at the specified index.

    """

    # Particle properties
    name: str
    pt: np.ndarray = None
    eta: np.ndarray = None
    phi: np.ndarray = None
    mass: np.ndarray = None
    pdgId: np.ndarray = None
    particle_jet_idx: np.ndarray = None

    # Additional properties
    classId: np.ndarray = None
    n: int = None
    charge: np.ndarray = None
    status: np.ndarray = None
    jets: list = None

    def __post_init__(self):
        self.n = np.array([len(el) for el in self.pt])
        if self.status is not None:
            return
        if self.pdgId is not None:
            self.classId = np.array(
                [
                    np.array([pdgid_class_dict[el] for el in pdgid], dtype=np.int32)
                    for pdgid in self.pdgId
                ],
                dtype=object,
            )
            self.charge = np.array(
                [
                    np.array([np.sign(pdgid) for pdgid in pdgid], dtype=np.int32)
                    for pdgid in self.pdgId
                ],
                dtype=object,
            )

    def __len__(self):
        return len(self.n)

    def __repr__(self) -> str:
        return f"{self.name}"

    def __getitem__(self, idx):
        assert idx < len(self), f"Index {idx} out of range"
        if self.status is not None:
            return np.stack(
                [
                    self.pt[idx],
                    self.eta[idx],
                    self.phi[idx],
                    self.mass[idx],
                    self.pdgId[idx],
                    self.status[idx],
                ],
                axis=-1,
            )
        else:
            return np.stack(
                [
                    self.pt[idx],
                    self.eta[idx],
                    self.phi[idx],
                    self.mass[idx],
                    self.pdgId[idx],
                    self.classId[idx],
                    self.charge[idx],
                ],
                axis=-1,
            )

    def apply_filter(self, f):
        self.pt = np.array(
            [pt_ev[f_ev] for pt_ev, f_ev in zip(self.pt, f)], dtype=object
        )
        self.eta = np.array(
            [eta_ev[f_ev] for eta_ev, f_ev in zip(self.eta, f)], dtype=object
        )
        self.phi = np.array(
            [phi_ev[f_ev] for phi_ev, f_ev in zip(self.phi, f)], dtype=object
        )
        if self.mass is not None:
            self.mass = np.array(
                [mass_ev[f_ev] for mass_ev, f_ev in zip(self.mass, f)], dtype=object
            )
        if self.pdgId is not None:
            self.pdgId = np.array(
                [pdgId_ev[f_ev] for pdgId_ev, f_ev in zip(self.pdgId, f)], dtype=object
            )
        if self.status is not None:
            self.status = np.array(
                [status_ev[f_ev] for status_ev, f_ev in zip(self.status, f)],
                dtype=object,
            )
        if self.classId is not None:
            self.classId = np.array(
                [classId_ev[f_ev] for classId_ev, f_ev in zip(self.classId, f)],
                dtype=object,
            )
        if self.charge is not None:
            self.charge = np.array(
                [charge_ev[f_ev] for charge_ev, f_ev in zip(self.charge, f)],
                dtype=object,
            )
        if self.particle_jet_idx is not None:
            self.particle_jet_idx = np.array(
                [pji_ev[f_ev] for pji_ev, f_ev in zip(self.particle_jet_idx, f)],
                dtype=object,
            )
        self.n = np.array([len(pt_ev) for pt_ev in self.pt])

    def get_4vectors(self):
        if self.mass is None:
            print(
                f"\033[93mWarning: mass is not defined for particle collection {self.name}! Assuming mass=0 for jet clustering...\033[0m"
            )
            mass = [np.zeros_like(pt_ev) for pt_ev in self.pt]
        else:
            mass = self.mass

        four_vectors = []

        for pt_ev, eta_ev, phi_ev, mass_ev in zip(self.pt, self.eta, self.phi, mass):
            four_vectors_ev = vec.array(
                {"pt": pt_ev, "phi": phi_ev, "eta": eta_ev, "mass": mass_ev}
            )

            four_vectors.append(four_vectors_ev)

        return four_vectors

    def cluster_jets(self, algo, radius, pt_min=8, n_const_min=2, eta_max=2.5):
        self.jets = []
        self.particle_jet_idx = np.array(
            [np.full(len(pt_ev), -999).astype(int) for pt_ev in self.pt], dtype=object
        )
        four_vectors = self.get_4vectors()

        print(
            f"Jets clustered using {algo} algorithm (R={radius}) and required to have pt > {pt_min} GeV, |eta| < {eta_max}, and Nconstituents >= {n_const_min}"
        )

        for ev_idx, four_vectors_ev in enumerate(tqdm.tqdm(four_vectors)):
            user_indices_ev = np.arange(len(four_vectors_ev))
            jets_ev = get_pt_sorted_jets(
                four_vectors_ev,
                algo,
                radius,
                pt_min,
                n_const_min,
                eta_max,
                user_indices_ev,
            )
            self.jets.append(jets_ev)

            # Set jet index for each clustered particle
            # Loop through pt-sorted jets backwards (higher-pt jets overwrite lower-pt ones)
            for i, jet in enumerate(jets_ev[::-1]):
                jet_idx = len(jets_ev) - 1 - i
                self.particle_jet_idx[ev_idx][jet.constituents_idx] = jet_idx


@dataclass
class GenJetCollection:
    "Class storing information about generic jet collection"

    # Jet properties
    name: str = ""
    pt: np.ndarray = None
    eta: np.ndarray = None
    phi: np.ndarray = None
    mass: np.ndarray = None

    hard_pid: np.ndarray = None
    hard_pt: np.ndarray = None
    hard_y: np.ndarray = None
    hard_phi: np.ndarray = None

    jec: np.ndarray = None
    weight: np.ndarray = None

    def __len__(self):
        return len(self.pt)

    def __repr__(self) -> str:
        return f"{self.name} with {len(self)} jets"


@dataclass
class GenData:
    "Class storing information about generic HEP data"

    # PFlowParticles
    pflowParticles: GenParticleCollection = None
    # TruthParticles
    truthParticles: GenParticleCollection = None
    # FastSimParticles
    fsParticles: GenParticleCollection = None

    # TruthJets
    truthJets: GenJetCollection = None
    # PFlowJets
    pflowJets: GenJetCollection = None
    pflowJets_rc: GenJetCollection = None
    # FastSimJets
    fsJets: GenJetCollection = None
    fsJets_rc: GenJetCollection = None

    def load_delphes(self, filepath, n_events=None, pfcand=True):
        "Load Delphes data from a given file"

        # Open file
        with uproot.open(filepath) as file:
            tree = file["Delphes"]
            # Load PFlowParticles
            if pfcand:
                print("Loading Delphes PFlowParticles...")
                self.pflowParticles = GenParticleCollection(
                    name="PFlowParticles",
                    pt=tree["ParticleFlowCandidate.PT"].array(
                        library="np", entry_stop=n_events
                    ),
                    eta=tree["ParticleFlowCandidate.Eta"].array(
                        library="np", entry_stop=n_events
                    ),
                    phi=tree["ParticleFlowCandidate.Phi"].array(
                        library="np", entry_stop=n_events
                    ),
                    mass=tree["ParticleFlowCandidate.Mass"].array(
                        library="np", entry_stop=n_events
                    ),
                    pdgId=tree["ParticleFlowCandidate.PID"].array(
                        library="np", entry_stop=n_events
                    ),
                )
            else:
                print("Loading Delphes Tracks and Towers...")
                track_pt = tree["Track.PT"].array(library="np", entry_stop=n_events)
                track_eta = tree["Track.Eta"].array(library="np", entry_stop=n_events)
                track_phi = tree["Track.Phi"].array(library="np", entry_stop=n_events)
                # track_mass = tree['Track.Mass'].array(library="np", entry_stop=n_events)
                # track_pid = tree['Track.PID'].array(library="np", entry_stop=n_events)

                tower_pt = tree["Tower.ET"].array(library="np", entry_stop=n_events)
                tower_eta = tree["Tower.Eta"].array(library="np", entry_stop=n_events)
                tower_phi = tree["Tower.Phi"].array(library="np", entry_stop=n_events)
                # tower_mass = tree['Tower.Mass'].array(library="np", entry_stop=n_events)
                # tower_pid = tree['Tower.PID'].array(library="np", entry_stop=n_events)

                pf_pt = []
                pf_eta = []
                pf_phi = []
                if n_events is None:
                    n_events = len(track_pt)
                for i in range(n_events):
                    pt = np.concatenate([track_pt[i], tower_pt[i]])
                    eta = np.concatenate([track_eta[i], tower_eta[i]])
                    phi = np.concatenate([track_phi[i], tower_phi[i]])
                    # mass = np.concatenate([track_mass[i], tower_mass[i]])
                    # pid = np.concatenate([track_pid[i], tower_pid[i]])

                    pf_pt.append(pt)
                    pf_eta.append(eta)
                    pf_phi.append(phi)
                    # pf_mass.append(mass)
                    # pf_pid.append(pid)

                self.pflowParticles = GenParticleCollection(
                    name="PFlowParticles",
                    pt=np.array(pf_pt, dtype=object),
                    eta=np.array(pf_eta, dtype=object),
                    phi=np.array(pf_phi, dtype=object),
                    # mass=np.array(pf_mass, dtype=object),
                    # pdgId=np.array(pf_pid, dtype=object),
                )

            # Load TruthParticles
            print("Loading Delphes TruthParticles...")
            self.truthParticles = GenParticleCollection(
                name="TruthParticles",
                pt=tree["Particle.PT"].array(library="np", entry_stop=n_events),
                eta=tree["Particle.Eta"].array(library="np", entry_stop=n_events),
                phi=tree["Particle.Phi"].array(library="np", entry_stop=n_events),
                mass=tree["Particle.Mass"].array(library="np", entry_stop=n_events),
                pdgId=tree["Particle.PID"].array(library="np", entry_stop=n_events),
            )

            # Load TruthJets
            print("Loading Delphes TruthJets...")
            self.truthJets = GenJetCollection(
                name="TruthJets",
                pt=np.concatenate(
                    tree["GenJet.PT"].array(library="np", entry_stop=n_events)
                ),
                eta=np.concatenate(
                    tree["GenJet.Eta"].array(library="np", entry_stop=n_events)
                ),
                phi=np.concatenate(
                    tree["GenJet.Phi"].array(library="np", entry_stop=n_events)
                ),
                mass=np.concatenate(
                    tree["GenJet.Mass"].array(library="np", entry_stop=n_events)
                ),
            )

            # Load PFlowJets
            print("Loading Delphes PFlowJets...")
            self.pflowJets = GenJetCollection(
                name="PFlowJets",
                pt=np.concatenate(
                    tree["Jet.PT"].array(library="np", entry_stop=n_events)
                ),
                eta=np.concatenate(
                    tree["Jet.Eta"].array(library="np", entry_stop=n_events)
                ),
                phi=np.concatenate(
                    tree["Jet.Phi"].array(library="np", entry_stop=n_events)
                ),
                mass=np.concatenate(
                    tree["Jet.Mass"].array(library="np", entry_stop=n_events)
                ),
            )

    def load_cms(self, data, n_events=None, entry_start=0, entry_stop=None):
        "Load CMS data from a given file"
        gens = data.gens
        pfcs = data.pfcs
        if n_events is None:
            n_events = len(gens)
        if entry_stop is None:
            entry_stop = min(entry_start + n_events, len(gens))
        # Load PFlowParticles
        print("Loading CMS PFlowParticles...")
        self.pflowParticles = GenParticleCollection(
            name="PFlowParticles",
            pt=np.array(
                [pfc[:, 0] for pfc in pfcs[entry_start:entry_stop]], dtype=object
            ),
            eta=np.array(
                [pfc[:, 1] for pfc in pfcs[entry_start:entry_stop]], dtype=object
            ),
            phi=np.array(
                [normalize_phi(pfc[:, 2]) for pfc in pfcs[entry_start:entry_stop]],
                dtype=object,
            ),
            mass=np.array(
                [pfc[:, 3] for pfc in pfcs[entry_start:entry_stop]], dtype=object
            ),
            pdgId=np.array(
                [pfc[:, 4] for pfc in pfcs[entry_start:entry_stop]], dtype=object
            ),
        )
        # Load TruthParticles
        print("Loading CMS TruthParticles...")
        self.truthParticles = GenParticleCollection(
            name="TruthParticles",
            pt=np.array(
                [gen[:, 0] for gen in gens[entry_start:entry_stop]], dtype=object
            ),
            eta=np.array(
                [gen[:, 1] for gen in gens[entry_start:entry_stop]], dtype=object
            ),
            phi=np.array(
                [normalize_phi(gen[:, 2]) for gen in gens[entry_start:entry_stop]],
                dtype=object,
            ),
            mass=np.array(
                [gen[:, 3] for gen in gens[entry_start:entry_stop]], dtype=object
            ),
            pdgId=np.array(
                [gen[:, 4] for gen in gens[entry_start:entry_stop]], dtype=object
            ),
        )

        # Load TruthJets
        print("Loading CMS TruthJets...")
        self.truthJets = GenJetCollection(
            name="TruthJets",
            pt=data.gen_jet_pts[entry_start:entry_stop],
            eta=data.gen_jet_etas[entry_start:entry_stop],
            phi=normalize_phi(data.gen_jet_phis[entry_start:entry_stop]),
            mass=data.gen_jet_ms[entry_start:entry_stop],
            hard_pid=data.hard_pids[entry_start:entry_stop],
            hard_pt=data.hard_pts[entry_start:entry_stop],
            hard_y=data.hard_ys[entry_start:entry_stop],
            hard_phi=normalize_phi(data.hard_phis[entry_start:entry_stop]),
            jec=data.jecs[entry_start:entry_stop],
            weight=data.weights[entry_start:entry_stop],
        )

        # Load PFlowJets
        print("Loading CMS PFlowJets...")
        self.pflowJets = GenJetCollection(
            name="PFlowJets",
            pt=data.jet_pts[entry_start:entry_stop],
            eta=data.jet_etas[entry_start:entry_stop],
            phi=normalize_phi(data.jet_phis[entry_start:entry_stop]),
            mass=data.jet_ms[entry_start:entry_stop],
        )

    def load_pu(self, filepath, n_events=None):
        with uproot.open(filepath) as file:
            tr_tree = file["genparticles/Events"]

            print("Loading pileup TruthParticles...")
            self.truthParticles = GenParticleCollection(
                name="GenParticles",
                pt=tr_tree["GenPart_pt"].array(library="np", entry_stop=n_events),
                eta=tr_tree["GenPart_eta"].array(library="np", entry_stop=n_events),
                phi=tr_tree["GenPart_phi"].array(library="np", entry_stop=n_events),
                mass=tr_tree["GenPart_mass"].array(library="np", entry_stop=n_events),
                pdgId=tr_tree["GenPart_pdgId"].array(library="np", entry_stop=n_events),
                status=tr_tree["GenPart_status"].array(
                    library="np", entry_stop=n_events
                ),
            )

    def load_fs(self, filepaths, n_events=None, dr_cut=False, flow=False):
        def convert_class(p_class):
            p_class_new = np.zeros_like(p_class)
            p_class_new[p_class <= 2] = 1
            p_class_new[p_class > 2] = 0
            return p_class_new

        if isinstance(filepaths, str):
            filepaths = [filepaths]
        pf_data_all = []
        fs_data_all = []
        tr_data_all = []
        jet_data_all = []
        for path in filepaths:
            print(f"Loading FastSim data from {path}...")
            with uproot.open(path) as file:
                pf_tree = file["pflow_tree"]
                fs_tree = file["fastsim_tree"]
                tr_tree = file["truth_tree"]

                pf_pt_all = pf_tree["pf_pt"].array(library="np", entry_stop=n_events)
                fs_pt_all = fs_tree["fs_pt"].array(library="np", entry_stop=n_events)
                tr_pt_all = tr_tree["tr_pt"].array(library="np", entry_stop=n_events)

                pf_eta_all = pf_tree["pf_eta"].array(library="np", entry_stop=n_events)
                fs_eta_all = fs_tree["fs_eta"].array(library="np", entry_stop=n_events)
                tr_eta_all = tr_tree["tr_eta"].array(library="np", entry_stop=n_events)

                pf_phi_all = pf_tree["pf_phi"].array(library="np", entry_stop=n_events)
                fs_phi_all = fs_tree["fs_phi"].array(library="np", entry_stop=n_events)
                tr_phi_all = tr_tree["tr_phi"].array(library="np", entry_stop=n_events)

                if flow:
                    tr_ind_all = file["truth_tree"]["tr_ind"].array(
                        library="np", entry_stop=n_events
                    )
                    pf_ind_all = file["pflow_tree"]["pf_ind"].array(
                        library="np", entry_stop=n_events
                    )
                    fs_ind_all = file["fastsim_tree"]["fs_ind"].array(
                        library="np", entry_stop=n_events
                    )
                    (
                        pf_pt_all,
                        pf_eta_all,
                        pf_phi_all,
                        fs_pt_all,
                        fs_eta_all,
                        fs_phi_all,
                        tr_pt_all,
                        tr_eta_all,
                        tr_phi_all,
                    ) = convert_flow(
                        pf_pt_all,
                        pf_eta_all,
                        pf_phi_all,
                        pf_ind_all,
                        fs_pt_all,
                        fs_eta_all,
                        fs_phi_all,
                        fs_ind_all,
                        tr_pt_all,
                        tr_eta_all,
                        tr_phi_all,
                        tr_ind_all,
                    )
                pf_data = {
                    "pt": pf_pt_all / 1000,
                    "eta": pf_eta_all,
                    "phi": pf_phi_all,
                }
                fs_data = {
                    "pt": fs_pt_all / 1000,
                    "eta": fs_eta_all,
                    "phi": fs_phi_all,
                }
                tr_data = {
                    "pt": tr_pt_all / 1000,
                    "eta": tr_eta_all,
                    "phi": tr_phi_all,
                }
                if "jet_tree" in [el.split(";")[0] for el in file.keys()]:
                    print("Loading jets")
                    jet_dict = {
                        "truth_jet_pt": "pt",
                        "truth_jet_eta": "eta",
                        "truth_jet_phi": "phi",
                        "pid": "hard_pid",
                        "hard_pt": "hard_pt",
                        "hard_eta": "hard_y",
                        "hard_phi": "hard_phi",
                        "jec": "jec",
                        "weight": "weight",
                    }
                    jet_data = {}
                    if "truth_jet_pt" not in file["jet_tree"].keys():
                        jet_data = None
                    else:
                        for branch_name, new_name in jet_dict.items():
                            jet_data[new_name] = file["jet_tree"][branch_name].array(
                                library="np", entry_stop=n_events
                            )
                else:
                    jet_data = None
                pf_data_all.append(pf_data)
                fs_data_all.append(fs_data)
                tr_data_all.append(tr_data)
                jet_data_all.append(jet_data)

        self.pflowParticles = GenParticleCollection(
            name="PFlowParticles",
            pt=np.concatenate([el["pt"] for el in pf_data_all]),
            eta=np.concatenate([el["eta"] for el in pf_data_all]),
            phi=np.concatenate([el["phi"] for el in pf_data_all]),
        )
        self.fsParticles = GenParticleCollection(
            name="FastSimParticles",
            pt=np.concatenate([el["pt"] for el in fs_data_all]),
            eta=np.concatenate([el["eta"] for el in fs_data_all]),
            phi=np.concatenate([el["phi"] for el in fs_data_all]),
        )
        self.truthParticles = GenParticleCollection(
            name="TruthParticles",
            pt=np.concatenate([el["pt"] for el in tr_data_all]),
            eta=np.concatenate([el["eta"] for el in tr_data_all]),
            phi=np.concatenate([el["phi"] for el in tr_data_all]),
        )
        if jet_data_all[0] is not None:
            self.truthJets = GenJetCollection(
                name="TruthJets",
                pt=np.concatenate([el["pt"] for el in jet_data_all]),
                eta=np.concatenate([el["eta"] for el in jet_data_all]),
                phi=np.concatenate([el["phi"] for el in jet_data_all]),
                hard_pid=np.concatenate([el["hard_pid"] for el in jet_data_all]),
                hard_pt=np.concatenate([el["hard_pt"] for el in jet_data_all]),
                hard_y=np.concatenate([el["hard_y"] for el in jet_data_all]),
                hard_phi=np.concatenate([el["hard_phi"] for el in jet_data_all]),
                jec=np.concatenate([el["jec"] for el in jet_data_all]),
                weight=np.concatenate([el["weight"] for el in jet_data_all]),
            )


def leading_jet_filter(particle_collection, radius=0.5, name="LeadingJets"):
    ### clustering first
    particle_collection.cluster_jets("antikt", radius)

    ### keep only particles in leading jet
    particle_in_leading_jet_mask = [
        (particle_jet_idx_ev == 0).astype(bool)
        for particle_jet_idx_ev in particle_collection.particle_jet_idx
    ]
    particle_collection.apply_filter(particle_in_leading_jet_mask)

    ### build a jet collection
    leading_jet_pt, leading_jet_eta, leading_jet_phi, leading_jet_mass = [], [], [], []
    for jets_ev in particle_collection.jets:
        leading_jet_pt.append(jets_ev[0].pt() if len(jets_ev) > 0 else -999)
        leading_jet_eta.append(jets_ev[0].eta() if len(jets_ev) > 0 else -999)
        leading_jet_phi.append(jets_ev[0].phi_std() if len(jets_ev) > 0 else -999)
        leading_jet_mass.append(jets_ev[0].m() if len(jets_ev) > 0 else -999)

    jet_collection = GenJetCollection(
        name=name,
        pt=np.array(leading_jet_pt),
        eta=np.array(leading_jet_eta),
        phi=np.array(leading_jet_phi),
        mass=np.array(leading_jet_mass),
    )

    return particle_collection, jet_collection
