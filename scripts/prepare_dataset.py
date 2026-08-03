#!/usr/bin/env python3
"""Build a class-balanced subset of an IBM synthetic AML transaction file.

The raw files are heavily imbalanced (laundering is well under 1% of rows), so
the reported experiments train on a subset that keeps every laundering
transaction and samples clean ones down to a target ratio.

    python scripts/prepare_dataset.py data/HI-Large_Trans.csv \
        --out data/balanced.csv --fraud-ratio 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from amlgnn.cli import setup_logging

log = logging.getLogger("prepare_dataset")


def balance(
    transactions: pd.DataFrame,
    fraud_ratio: float = 0.5,
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Down-sample clean transactions until laundering is ``fraud_ratio`` of rows."""
    if not 0 < fraud_ratio < 1:
        raise ValueError("fraud_ratio must be strictly between 0 and 1")

    fraud = transactions[transactions["Is Laundering"] == 1]
    clean = transactions[transactions["Is Laundering"] == 0]
    if fraud.empty:
        raise ValueError("no laundering transactions found")

    wanted_clean = int(len(fraud) * (1 - fraud_ratio) / fraud_ratio)
    if wanted_clean > len(clean):
        log.warning(
            "only %d clean transactions available, wanted %d; ratio will be %.3f",
            len(clean),
            wanted_clean,
            len(fraud) / (len(fraud) + len(clean)),
        )
        wanted_clean = len(clean)

    balanced = pd.concat([fraud, clean.sample(n=wanted_clean, random_state=seed)])
    if max_rows is not None and len(balanced) > max_rows:
        balanced = balanced.sample(n=max_rows, random_state=seed)

    # Chronological order keeps the per-account time gaps meaningful.
    return balanced.sort_values("Timestamp").reset_index(drop=True)


def summarise(transactions: pd.DataFrame, title: str) -> None:
    accounts = pd.concat([transactions["Account"], transactions["Account.1"]]).nunique()
    fraud = int(transactions["Is Laundering"].sum())
    log.info(
        "%s: %d transactions, %d accounts, %d laundering (%.3f%%)",
        title,
        len(transactions),
        accounts,
        fraud,
        100 * fraud / max(len(transactions), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="raw IBM AML transaction CSV")
    parser.add_argument("--out", type=Path, required=True, help="where to write the balanced CSV")
    parser.add_argument("--fraud-ratio", type=float, default=0.5, help="target share of laundering rows")
    parser.add_argument("--max-rows", type=int, default=None, help="cap on the output size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log.info("reading %s", args.source)
    transactions = pd.read_csv(args.source, parse_dates=["Timestamp"])
    summarise(transactions, "source")

    balanced = balance(transactions, args.fraud_ratio, args.max_rows, args.seed)
    summarise(balanced, "balanced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(args.out, index=False)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
