from pathlib import Path

import numpy as np
import pytest

from parnassus_core.data.loaders import (
    LUNDFileLoader,
    ROOTFileLoader,
    is_lund_path,
    open_file_loader,
    resolve_lund_files,
)
from parnassus_core.data.loaders.lund_loader import read_lund_file

# A header with the 10 required fields and two optional ones.
HEADER = "{n} 1 1 0.0 1.0 11 10.600 2212 1 0.68E+05 7 8"


def _particle(index, ptype, pid, px, py, pz, mass, vz=0.0, parent=0):
    return (
        f"{index} 0. {ptype} {pid} {parent} 0 {px} {py} {pz} 0.0 {mass} 0.01 -0.02 {vz}"
    )


# Event 0: beam rows (type 21), an intermediate (type 13), and four final-state
# particles of which one is an electron neutrino and one lies below 1 degree.
EVENT_0 = [
    HEADER.format(n=7),
    _particle(1, 21, 11, 0.0, 0.0, 10.6, 0.0005),
    _particle(2, 21, 2212, 0.0, 0.0, 0.0, 0.9383),
    _particle(3, 13, 2101, 0.1, 0.2, 0.3, 0.5793),
    _particle(4, 1, 11, -0.4, -1.2, 7.2, 0.0005, parent=1),
    _particle(5, 1, 211, 0.3, 0.0, 0.7, 0.1396, vz=1.5),
    _particle(6, 1, 12, 0.2, 0.1, 1.0, 0.0),
    _particle(7, 1, 22, 0.001, 0.0, 5.0, 0.0),
]
# Event 1: every final-state particle is cut (theta < 1 degree).
EVENT_1 = [
    HEADER.format(n=2),
    _particle(1, 21, 11, 0.0, 0.0, 10.6, 0.0005),
    _particle(2, 1, 22, 0.0, 0.001, 3.0, 0.0),
]
# Event 2: two particles, like event 3 (equal-length events).
EVENT_2 = [
    HEADER.format(n=2),
    _particle(1, 1, 11, 0.5, 0.5, 4.0, 0.0005),
    _particle(2, 1, 2112, -0.3, 0.4, 1.0, 0.9396),
]
EVENT_3 = [
    HEADER.format(n=2),
    _particle(1, 1, -211, 0.1, -0.4, 2.0, 0.1396),
    _particle(2, 1, 22, 0.2, 0.2, 0.2, 0.0),
]


def _write(path: Path, events: list[list[str]], blank_lines: bool = False) -> Path:
    separator = "\n\n" if blank_lines else "\n"
    path.write_text(separator.join(line for event in events for line in event) + "\n")
    return path


@pytest.fixture
def lund_file(tmp_path):
    return _write(tmp_path / "sample.dat", [EVENT_0, EVENT_1, EVENT_2, EVENT_3])


def test_selection_keeps_final_state_particles(lund_file):
    events = read_lund_file(lund_file)
    assert events.counts.tolist() == [2, 0, 2, 2]
    assert events.particles["pdgId"].tolist() == [11, 211, 11, 2112, -211, 22]


def test_kinematics_from_momentum_and_mass_columns(lund_file):
    events = read_lund_file(lund_file)
    px, py, pz, mass = 0.3, 0.0, 0.7, 0.1396  # the pi+ of event 0
    p = np.sqrt(px**2 + py**2 + pz**2)
    particles = {var: arr[1] for var, arr in events.particles.items()}
    assert particles["p"] == pytest.approx(p, rel=1e-6)
    assert particles["theta"] == pytest.approx(
        np.arctan2(np.hypot(px, py), pz), rel=1e-6
    )
    assert particles["phi"] == pytest.approx(0.0, abs=1e-7)
    assert particles["beta"] == pytest.approx(p / np.sqrt(mass**2 + p**2), rel=1e-6)
    assert particles["vz"] == pytest.approx(1.5)
    assert particles["vx"] == pytest.approx(0.01)
    assert "pt" not in events.particles


def test_loader_branches_match_eager_dataset_layout(lund_file):
    loader = LUNDFileLoader(lund_file)
    assert loader.get_total_events() == 4
    arrays = loader.load_branches(["ntruth", "eventNumber", "truth_p"], 0, 4)
    assert arrays["ntruth"].tolist() == [2, 0, 2, 2]
    assert arrays["eventNumber"].tolist() == [0, 1, 2, 3]
    # Equal-length events must still come back as one array per event.
    assert arrays["truth_p"].dtype == object
    assert arrays["truth_p"].shape == (4,)
    assert [len(event) for event in arrays["truth_p"]] == [2, 0, 2, 2]


def test_iter_branches_matches_load_branches(lund_file):
    loader = LUNDFileLoader(lund_file)
    names = ["ntruth", "eventNumber", "truth_pdgId"]
    chunks = list(loader.iter_branches(names, 1, None, 2))
    assert [len(chunk["ntruth"]) for chunk in chunks] == [2, 1]
    bulk = loader.load_branches(names, 1, 4)
    assert np.concatenate([c["eventNumber"] for c in chunks]).tolist() == [1, 2, 3]
    joined = np.concatenate([c["truth_pdgId"] for c in chunks])
    assert [event.tolist() for event in joined] == [
        event.tolist() for event in bulk["truth_pdgId"]
    ]


def test_event_numbers_run_across_files_and_blank_lines_are_skipped(tmp_path):
    _write(tmp_path / "a_0000.dat", [EVENT_0, EVENT_1], blank_lines=True)
    _write(tmp_path / "a_0001.dat", [EVENT_2, EVENT_3])
    for source in (tmp_path, tmp_path / "a_*.dat"):
        loader = LUNDFileLoader(source)
        arrays = loader.load_branches(["ntruth", "eventNumber"], 0, 4)
        assert arrays["ntruth"].tolist() == [2, 0, 2, 2]
        assert arrays["eventNumber"].tolist() == [0, 1, 2, 3]


def test_max_events_stops_parsing(tmp_path):
    _write(tmp_path / "a_0000.dat", [EVENT_0, EVENT_1])
    second = _write(tmp_path / "a_0001.dat", [EVENT_2, EVENT_3])
    loader = LUNDFileLoader(tmp_path, max_events=1)
    assert loader.get_total_events() == 1
    second.write_text("not a LUND file\n")  # never reached
    assert loader.load_branches(["ntruth"], 0, 5)["ntruth"].tolist() == [2]


def test_lazy_iteration_parses_only_what_is_consumed(tmp_path):
    _write(tmp_path / "a_0000.dat", [EVENT_0, EVENT_1])
    _write(tmp_path / "a_0001.dat", ["garbage line".split()])
    chunks = LUNDFileLoader(tmp_path).iter_branches(["ntruth"], 0, None, 2)
    assert next(chunks)["ntruth"].tolist() == [2, 0]
    with pytest.raises(ValueError, match="a_0001.dat:1"):
        next(chunks)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([HEADER.format(n=2), _particle(1, 1, 11, 0.1, 0.1, 1.0, 0.0)], "truncated"),
        ([HEADER.format(n=1), "1 0. 1 11 0 0 0.1 0.1 1.0"], "sample.dat:2"),
        (["3 1 1 0.0"], "event header"),
    ],
)
def test_malformed_input_names_the_line(tmp_path, lines, message):
    path = _write(tmp_path / "sample.dat", [lines])
    with pytest.raises(ValueError, match=message):
        read_lund_file(path)


def test_reconstructed_branches_are_rejected(lund_file):
    with pytest.raises(KeyError, match="eval mode only"):
        LUNDFileLoader(lund_file).load_branches(["pflow_p"], 0, 1)


def test_dispatch_by_extension(tmp_path, lund_file):
    assert isinstance(open_file_loader(lund_file), LUNDFileLoader)
    assert isinstance(open_file_loader(tmp_path / "*.dat"), LUNDFileLoader)
    assert is_lund_path(tmp_path) and not is_lund_path(tmp_path / "x.root")
    with pytest.raises(ValueError, match="Unsupported input"):
        open_file_loader(tmp_path / "data.h5")
    with pytest.raises(FileNotFoundError):
        resolve_lund_files(tmp_path / "missing_*.dat")


def test_root_loader_selected_for_root_files(tmp_path):
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "events.root"
    with uproot.recreate(path) as file:
        file.mktree("evt_tree", {"eventNumber": np.int32})
        file["evt_tree"].extend({"eventNumber": np.arange(3, dtype=np.int32)})
    loader = open_file_loader(path)
    assert isinstance(loader, ROOTFileLoader)
    assert loader.has_branch("eventNumber") and not loader.has_branch("truth_p")
    chunks = list(loader.iter_branches(["eventNumber"], 0, None, 2))
    assert [chunk["eventNumber"].tolist() for chunk in chunks] == [[0, 1], [2]]
