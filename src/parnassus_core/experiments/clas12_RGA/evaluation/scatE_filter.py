"""Write the full-simulation reference for the CLAS12 evaluation chain.

The fast simulation is evaluated on events whose scattered electron was
reconstructed.  The fast-sim side gets that selection from the scatE classifier,
applied inside ``evaluation/eval.py`` (the ``scatE_filter:`` block of the eval
config, see :mod:`parnassus_core.evaluation.event_filter`).  The full-simulation
reference instead uses the true label: this script keeps the events of a
CLAS12 ``AllEvents`` ROOT file with ``pflow_scatE == 1``::

    AllEvents.root -> eval.py (scatE_filter: sample) -> postprocess_eval.py
                   -> scatE_filter.py -> <stem>_truth.root ----^  (-d reference)

The output is the same flat ``evt_tree`` layout as the input, with every source
branch copied verbatim, so ``postprocess_eval`` joins it to the eval output by
``eventNumber``.  Read at least as many input events as the eval run scanned,
so the reference covers every event the fast simulation was offered.

Example
-------
::

    python -m parnassus_core.experiments.clas12_RGA.evaluation.scatE_filter \
        -i ../parnassus_data/v8/v8_test.root \
        -o ../parnassus_data/v8/v8_test_filtered_truth.root
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from rich import print as rprint
from tqdm import tqdm

from parnassus_core.evaluation.event_filter import LABEL_BRANCH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the events of a CLAS12 AllEvents ROOT file whose "
        "scattered electron was reconstructed (pflow_scatE == 1)."
    )
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--tree", type=str, default="evt_tree")
    parser.add_argument(
        "-ne",
        "--n-events",
        type=int,
        default=None,
        help="Input events to read (not output events). Default (or -1): the whole file.",
    )
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("--output must differ from --input")

    tree = uproot.open(f"{input_path}:{args.tree}")
    branches = list(tree.keys())
    for required in (LABEL_BRANCH, "eventNumber"):
        if required not in branches:
            raise KeyError(f"branch '{required}' missing from {input_path}")

    entry_start = args.entry_start
    entry_stop = tree.num_entries
    # A negative count means the whole file; passed through, -1 would become an
    # uproot entry_stop of -1, which silently drops the last event.
    if args.n_events is not None and args.n_events >= 0:
        entry_stop = min(entry_stop, entry_start + args.n_events)
    rprint(
        f"[bold blue]Reading:[/bold blue] {input_path} entries "
        f"[{entry_start}, {entry_stop}) of {tree.num_entries}"
    )

    n_read = 0
    n_kept = 0
    tree_created = False
    with uproot.recreate(output_path) as fout:
        progress = tqdm(total=entry_stop - entry_start, unit="evt", desc="Filtering")
        for chunk in tree.iterate(
            branches,
            step_size=args.chunk_size,
            entry_start=entry_start,
            entry_stop=entry_stop,
        ):
            keep = ak.to_numpy(chunk[LABEL_BRANCH]) > 0

            if not tree_created:
                # Declare the branches explicitly with mktree.  Plain
                # ``fout[name] = dict`` writes an RNTuple in uproot >= 5.7, but
                # the input is a TTree and downstream ROOT-based tooling expects
                # one, so the output must stay a TTree.  Types come from the
                # unfiltered chunk, so an all-rejected first chunk still gives a
                # correctly typed (empty) tree.
                #
                # mktree gives every jagged branch its own counter
                # (``ntruth_p``, ``npflow_theta``, ...) rather than sharing the
                # source's single ``ntruth`` / ``npflow``; uproot rejects a
                # shared counter.  The duplicates are dead weight but harmless:
                # ``ntruth`` / ``npflow`` are still copied verbatim, and every
                # reader here requests branches by exact name.
                fout.mktree(
                    args.tree, {name: ak.type(chunk[name]) for name in branches}
                )
                tree_created = True

            if keep.any():
                fout[args.tree].extend(  # type: ignore[union-attr]
                    {name: chunk[name][keep] for name in branches}
                )

            n_read += len(chunk)
            n_kept += int(np.count_nonzero(keep))
            progress.update(len(chunk))
        progress.close()

    if n_kept == 0:
        raise RuntimeError(
            f"no events with {LABEL_BRANCH} == 1 over entries "
            f"[{entry_start}, {entry_stop}) -- {output_path} holds an empty tree"
        )
    rprint(
        f"\n[bold green]Done:[/bold green] {n_kept} of {n_read} events "
        f"({n_kept / n_read:.5f}) -> {output_path.absolute()}"
    )


if __name__ == "__main__":
    main()
