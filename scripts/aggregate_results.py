#!/usr/bin/env python3
"""Summarise a runs.csv into a Markdown table.

Defaults to the committed 2024 record that RESULTS.md reports on. Point it at
``output/runs.csv`` to summarise runs you have made yourself.

    python scripts/aggregate_results.py --since 20240625153000 --markdown
    python scripts/aggregate_results.py --runs-csv output/runs.csv --markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amlgnn.config import ARCHIVED_RUNS_CSV

METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Loss"]
ORDER = ["DisMult", "DisMult-T", "DisMult-T+W", "ComplEx", "ComplEx-T", "ComplEx-T+W",
         "DistMult", "DistMult-T", "DistMult-T+W"]


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby("Model")[METRICS].agg(["mean", "std", "count"])
    present = [m for m in ORDER if m in grouped.index]
    return grouped.loc[present + [m for m in grouped.index if m not in present]]


def to_markdown(summary: pd.DataFrame) -> str:
    header = "| Variant | Runs | " + " | ".join(METRICS) + " |"
    divider = "| --- | ---: | " + " | ".join(["---:"] * len(METRICS)) + " |"
    lines = [header, divider]
    for model, row in summary.iterrows():
        cells = []
        for metric in METRICS:
            mean, std = row[(metric, "mean")], row[(metric, "std")]
            cells.append(f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f}")
        runs = int(row[(METRICS[0], "count")])
        lines.append(f"| {model} | {runs} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-csv", type=Path, default=ARCHIVED_RUNS_CSV)
    parser.add_argument("--since", type=int, default=None, help="keep runs with Timestamp >= this")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)
    if args.since is not None:
        runs = runs[runs["Timestamp"] >= args.since]
    if runs.empty:
        raise SystemExit("no runs matched")

    summary = summarise(runs)
    print(to_markdown(summary) if args.markdown else summary.round(4).to_string())


if __name__ == "__main__":
    main()
