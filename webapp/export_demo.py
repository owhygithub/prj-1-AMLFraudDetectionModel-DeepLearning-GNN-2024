#!/usr/bin/env python3
"""Export a trained graph and its checkpoints as JSON for the web demo.

The demo runs the real model in the browser, so this ships everything the
forward pass needs: the graph tensors, the learned weight matrices, and enough
of the original CSV to render a readable transaction table.

    python webapp/export_demo.py \
        --graph data/demo-graph.pt \
        --transactions data/demo.csv \
        --artifacts artifacts \
        --out webapp/public/data

Re-point ``--graph``/``--artifacts`` at a model trained on the real IBM data
and the demo picks it up with no other changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from amlgnn.cli import setup_logging
from amlgnn.data import split_edges
from amlgnn.metrics import evaluate
from amlgnn.models import AMLLinkScorer
from amlgnn.preprocessing import TransactionGraph

log = logging.getLogger("export_demo")

VARIANT_ORDER = ["DistMult", "DistMult-T", "DistMult-T+W", "ComplEx", "ComplEx-T", "ComplEx-T+W"]


def _round(array: np.ndarray | torch.Tensor, decimals: int) -> list:
    """Round to keep the JSON small. 4 dp is far below the model's noise floor."""
    values = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)
    return np.round(values.astype(float), decimals).tolist()


def export_graph(graph: TransactionGraph, transactions: pd.DataFrame, split) -> dict:
    """The tensors the browser needs to reproduce the forward pass exactly."""
    adjacency = graph.adjacency.coalesce().indices()
    split_code = np.zeros(graph.num_edges, dtype=int)
    split_code[split.val.numpy()] = 1
    split_code[split.test.numpy()] = 2

    return {
        "meta": {
            "accounts": graph.num_nodes,
            "transactions": graph.num_edges,
            "laundering": int(graph.data.y.sum()),
            "nodeFeatures": graph.data.x.size(1),
            "edgeFeatures": graph.data.edge_attr.size(1),
            "exported": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "accounts": [str(a) for a in graph.accounts],
        "nodeFeatureNames": graph.node_feature_names,
        "edgeFeatureNames": graph.edge_feature_names,
        "x": _round(graph.data.x, 6),
        "edgeIndex": graph.data.edge_index.tolist(),
        "edgeAttr": _round(graph.data.edge_attr, 6),
        "timeCloseness": _round(graph.time_closeness, 6),
        "label": graph.data.y.int().tolist(),
        "split": split_code.tolist(),
        "adjacency": {"row": adjacency[0].tolist(), "col": adjacency[1].tolist()},
        "transactions": {
            "amount": _round(transactions["Amount Paid"].to_numpy(), 2),
            "currency": transactions["Payment Currency"].astype(str).tolist(),
            "format": transactions["Payment Format"].astype(str).tolist(),
            "timestamp": pd.to_datetime(transactions["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M").tolist(),
        },
    }


def export_variant(checkpoint_path: Path, graph: TransactionGraph, split) -> dict:
    """Weights plus the metrics this checkpoint actually achieves on the test split."""
    payload = torch.load(checkpoint_path, weights_only=False)
    config = payload["config"]
    model = AMLLinkScorer(
        node_features=payload["node_features"],
        edge_features=payload["edge_features"],
        out_channels=config.out_channels,
        dropout=config.dropout,
        decoder=config.decoder,
        use_time=config.use_time,
        learn_time_weight=config.learn_time_weight,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with torch.no_grad():
        _, _, logits = model(
            graph.data.x,
            graph.data.edge_index,
            graph.data.edge_attr,
            graph.adjacency,
            graph.time_closeness if model.use_time else None,
        )
        scores = torch.sigmoid(logits)

    report = evaluate(scores[split.test], graph.data.y[split.test], config.threshold)
    log.info("%-14s %s", config.name, report)

    state = model.state_dict()
    return {
        "name": config.name,
        "decoder": config.decoder,
        "useTime": config.use_time,
        "learnTimeWeight": config.learn_time_weight,
        "outChannels": config.out_channels,
        "threshold": config.threshold,
        "weightNode": _round(state["encoder.weight_node"], 8),
        "weightEdge": _round(state["encoder.weight_edge"], 8),
        "timeWeight": float(state["time_weight"].item()) if "time_weight" in state else None,
        "metrics": {k: v for k, v in report.as_dict().items() if v is not None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, default=Path("data/demo-graph.pt"))
    parser.add_argument("--transactions", type=Path, default=Path("data/demo.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("webapp/public/data"))
    parser.add_argument("--seed", type=int, default=42, help="must match the seed used for training")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    graph = TransactionGraph.load(args.graph)
    transactions = pd.read_csv(args.transactions, parse_dates=["Timestamp"])
    if len(transactions) != graph.num_edges:
        raise SystemExit(
            f"{args.transactions} has {len(transactions)} rows but the graph has "
            f"{graph.num_edges} edges. They must come from the same build."
        )
    split = split_edges(graph.data.y, seed=args.seed)
    log.info("graph: %d accounts, %d transactions", graph.num_nodes, graph.num_edges)

    variants = []
    for name in VARIANT_ORDER:
        path = args.artifacts / f"{name}.pt"
        if not path.exists():
            log.warning("no checkpoint for %s at %s, skipping", name, path)
            continue
        variants.append(export_variant(path, graph, split))
    if not variants:
        raise SystemExit(f"no checkpoints found in {args.artifacts}")

    args.out.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("graph.json", export_graph(graph, transactions, split)),
        ("models.json", {"variants": variants}),
    ):
        path = args.out / filename
        path.write_text(json.dumps(payload, separators=(",", ":")))
        log.info("wrote %s (%.1f KB)", path, path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
