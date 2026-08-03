#!/usr/bin/env python3
"""Turn a transaction CSV into the cached graph artefact used for training.

    python scripts/build_graph.py data/balanced.csv --out data/graph.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlgnn.cli import setup_logging
from amlgnn.config import DATA_DIR
from amlgnn.preprocessing import build_graph, load_transactions

log = logging.getLogger("build_graph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="transaction CSV")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "graph.pt")
    parser.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    transactions = load_transactions(args.source, nrows=args.nrows)
    graph = build_graph(transactions)

    log.info(
        "graph: %d accounts, %d transactions, %d laundering",
        graph.num_nodes,
        graph.num_edges,
        int(graph.data.y.sum()),
    )
    log.info("saved %s", graph.save(args.out))


if __name__ == "__main__":
    main()
