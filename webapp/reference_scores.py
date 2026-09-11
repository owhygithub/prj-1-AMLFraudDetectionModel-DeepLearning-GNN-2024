#!/usr/bin/env python3
"""Score every transaction with PyTorch, for webapp/verify_model.mjs to check against.

    python webapp/reference_scores.py
    node webapp/verify_model.mjs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from amlgnn.models import AMLLinkScorer
from amlgnn.preprocessing import TransactionGraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, default=Path("data/demo-graph.pt"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("webapp/public/data/reference-scores.json"))
    args = parser.parse_args()

    graph = TransactionGraph.load(args.graph)
    scores = {}
    for path in sorted(args.artifacts.glob("*.pt")):
        payload = torch.load(path, weights_only=False)
        config = payload["config"]
        model = AMLLinkScorer(
            payload["node_features"],
            payload["edge_features"],
            config.out_channels,
            config.dropout,
            config.decoder,
            config.use_time,
            config.learn_time_weight,
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
        scores[config.name] = torch.sigmoid(logits).tolist()
        print(f"scored {config.name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
