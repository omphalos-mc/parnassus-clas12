import numpy as np
import pytest

from parnassus_core.data.loaders import FileLoader, FilteredFileLoader


def _ragged(parts):
    out = np.empty(len(parts), dtype=object)
    for index, part in enumerate(parts):
        out[index] = np.asarray(part, dtype=np.float32)
    return out


class InMemoryLoader(FileLoader):
    """Minimal FileLoader over in-memory branches, recording its reads."""

    def __init__(self, branches):
        super().__init__("memory")
        self.branches = branches
        self.iter_calls = []

    def has_branch(self, name):
        return name in self.branches

    def load_branches(self, branch_names, entry_start, entry_stop):
        return {
            name: self.branches[name][entry_start:entry_stop] for name in branch_names
        }

    def iter_branches(self, branch_names, entry_start, entry_stop, step_size):
        stop = len(self.branches["eventNumber"]) if entry_stop is None else entry_stop
        self.iter_calls.append((entry_start, stop))
        for lo in range(entry_start, stop, step_size):
            yield self.load_branches(branch_names, lo, min(lo + step_size, stop))

    def load_particle_collection(
        self, collection_name, variables, entry_start, entry_stop
    ):
        raise NotImplementedError

    def load_event_variables(self, variables, entry_start, entry_stop):
        return self.load_branches(variables, entry_start, entry_stop)

    def get_num_particles(self, collection_name, entry_start, entry_stop):
        return self.load_branches([f"n{collection_name}"], entry_start, entry_stop)[
            f"n{collection_name}"
        ]

    def get_total_events(self):
        return len(self.branches["eventNumber"])

    def close(self):
        pass


class EveryThird:
    """Keeps events whose eventNumber is divisible by 3; records what it saw."""

    branches = ("eventNumber",)
    optional_branches = ("label", "absent")

    def __init__(self):
        self.seen = []

    def __call__(self, chunk):
        self.seen.append(sorted(chunk))
        return chunk["eventNumber"] % 3 == 0


N_EVENTS = 20


@pytest.fixture
def inner():
    event_number = np.arange(N_EVENTS, dtype=np.int32)
    return InMemoryLoader(
        {
            "eventNumber": event_number,
            "label": event_number % 2,
            "truth_p": _ragged([[float(i)] * (i % 4) for i in range(N_EVENTS)]),
        }
    )


@pytest.mark.parametrize("chunk_size", [1, 4, 7, 100])
def test_reads_see_only_kept_events(inner, chunk_size):
    loader = FilteredFileLoader(inner, EveryThird(), chunk_size=chunk_size)
    kept = np.arange(0, N_EVENTS, 3)
    assert loader.get_total_events() == len(kept)

    arrays = loader.load_branches(["eventNumber", "truth_p"], 1, 5)
    assert arrays["eventNumber"].tolist() == kept[1:5].tolist()
    assert [len(event) for event in arrays["truth_p"]] == [i % 4 for i in kept[1:5]]

    chunks = list(loader.iter_branches(["eventNumber"], 0, None, 5))
    assert all(len(chunk["eventNumber"]) for chunk in chunks)
    streamed = np.concatenate([chunk["eventNumber"] for chunk in chunks])
    assert streamed.tolist() == kept.tolist()


def test_max_kept_truncates_within_a_chunk(inner):
    selector = EveryThird()
    loader = FilteredFileLoader(inner, selector, max_kept=2, chunk_size=8)
    # The first chunk (events 0-7) accepts 0, 3 and 6; only the first two count.
    assert loader.load_branches(["eventNumber"], 0, 10)["eventNumber"].tolist() == [
        0,
        3,
    ]
    assert len(selector.seen) == 1


def test_max_kept_stops_scanning(inner):
    selector = EveryThird()
    loader = FilteredFileLoader(inner, selector, max_kept=3, chunk_size=4)
    assert loader.get_total_events() == 3
    # Chunks 0-3 and 4-7 hold the first three kept events; the scan stops there.
    assert len(selector.seen) == 2


def test_optional_branches_only_when_present(inner):
    selector = EveryThird()
    FilteredFileLoader(inner, selector)
    assert selector.seen[0] == ["eventNumber", "label"]


def test_second_pass_reads_only_the_kept_span(inner):
    loader = FilteredFileLoader(inner, EveryThird(), chunk_size=100)
    list(loader.iter_branches(["eventNumber"], 2, 4, 3))
    assert inner.iter_calls[-1] == (6, 10)


def test_empty_selection(inner):
    class KeepNothing(EveryThird):
        def __call__(self, chunk):
            return np.zeros(len(chunk["eventNumber"]), dtype=bool)

    loader = FilteredFileLoader(inner, KeepNothing())
    assert loader.get_total_events() == 0
    assert loader.load_branches(["eventNumber"], 0, 5)["eventNumber"].tolist() == []
    assert list(loader.iter_branches(["eventNumber"], 0, None, 5)) == []


def test_mask_shape_is_checked(inner):
    class WrongShape(EveryThird):
        def __call__(self, chunk):
            return np.ones(1, dtype=bool)

    with pytest.raises(ValueError, match="mask of shape"):
        FilteredFileLoader(inner, WrongShape(), chunk_size=4)
