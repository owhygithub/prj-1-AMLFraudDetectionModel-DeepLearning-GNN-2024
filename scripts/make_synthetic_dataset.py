#!/usr/bin/env python3
"""Generate a transaction file with the IBM AML schema and planted typologies.

Real laundering is not random: it shows up as recognisable shapes in the
transaction graph. This generator lays down a background of ordinary payments
and then plants four of the classic ones, labelling only their edges as
laundering:

    fan-out    one account structures a sum out to many mules at once
    fan-in     many mules consolidate into a single collector
    cycle      funds move A -> B -> C -> A and come back washed
    chain      funds are layered forward through a line of accounts

Each pattern fires in a tight time window, which is what gives the ``-T``
model variants something to find that the plain ones cannot see.

It also plants *decoys*: payroll runs, merchant settlement and supply-chain
payments have the same burst-of-activity shape but are perfectly legitimate.
Without them the task is trivial. "Many transfers at once" separates the
classes on its own, every model saturates at p = 1.0, and nothing interesting
is left to compare. The decoys are what create borderline cases.

This is NOT the IBM dataset. It exists so the pipeline and the web demo can be
run end to end without a 1.5 GB download, and so that anything trained on it
is learning real graph structure rather than noise.

    python scripts/make_synthetic_dataset.py --out data/synthetic.csv

Use --rows/--accounts for a smoke test, or the defaults for something big
enough to train a demo model on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CURRENCIES = ["US Dollar", "Euro", "UK Pound"]
CURRENCY_P = [0.90, 0.07, 0.03]
FORMATS = ["Reinvestment", "Cheque", "Credit Card", "ACH", "Wire", "Cash"]
FORMAT_P = [0.36, 0.16, 0.20, 0.14, 0.09, 0.05]
#: Laundering leans towards instruments that move value fast, but only leans -
#: a format that gave the label away would make the graph structure redundant.
LAUNDERING_FORMATS = ["Cash", "Wire", "ACH", "Cheque", "Credit Card", "Reinvestment"]
LAUNDERING_FORMAT_P = [0.24, 0.22, 0.18, 0.14, 0.12, 0.10]

START = pd.Timestamp("2022-09-01")
WINDOW_SECONDS = 86_400 * 10


class _Builder:
    """Accumulates transaction rows as parallel lists."""

    def __init__(self, rng: np.random.Generator, accounts: int):
        self.rng = rng
        self.account_ids = [f"80{i:07X}" for i in range(accounts)]
        self.banks = rng.integers(0, 64, size=accounts)
        self.rows: list[dict] = []

    def add(self, source: int, target: int, amount: float, when: int, laundering: bool) -> None:
        if laundering:
            fmt = self.rng.choice(LAUNDERING_FORMATS, p=LAUNDERING_FORMAT_P)
        else:
            fmt = self.rng.choice(FORMATS, p=FORMAT_P)
        currency = self.rng.choice(CURRENCIES, p=CURRENCY_P)
        amount = round(float(amount), 2)
        self.rows.append(
            {
                "Timestamp": START + pd.Timedelta(seconds=int(when)),
                "From Bank": int(self.banks[source]),
                "Account": self.account_ids[source],
                "To Bank": int(self.banks[target]),
                "Account.1": self.account_ids[target],
                "Amount Received": amount,
                "Receiving Currency": currency,
                "Amount Paid": amount,
                "Payment Currency": currency,
                "Payment Format": fmt,
                "Is Laundering": int(laundering),
            }
        )


def _background(builder: _Builder, rows: int) -> None:
    """Ordinary payments: uniform pairs, lognormal amounts, spread over time."""
    rng = builder.rng
    n = len(builder.account_ids)
    for _ in range(rows):
        source, target = rng.integers(0, n, size=2)
        if source == target:
            continue
        builder.add(
            int(source),
            int(target),
            rng.lognormal(6.2, 1.6),
            rng.integers(0, WINDOW_SECONDS),
            laundering=False,
        )


def _fan_out(builder: _Builder, hub: int, mules: np.ndarray, start: int, laundering: bool = True) -> None:
    """Structuring: a lump sum broken into many sub-threshold transfers.

    The legitimate version of this shape is a payroll run. Same fan, but the
    amounts look like salaries rather than being pressed up against a reporting
    threshold, and they go out on a schedule rather than in a rush.
    """
    rng = builder.rng
    for offset, mule in enumerate(mules):
        if laundering:
            amount = rng.uniform(7_400, 9_900)  # kept under a round threshold
            gap = int(rng.integers(20, 240))
        else:
            amount = rng.uniform(2_100, 9_400)  # salaries, wider spread
            gap = int(rng.integers(40, 900))
        builder.add(hub, int(mule), amount, start + offset * gap, laundering)


def _fan_in(builder: _Builder, collector: int, mules: np.ndarray, start: int, laundering: bool = True) -> None:
    """Consolidation: the mules push their balances into one account.

    The legitimate version is a merchant settling the day's takings.
    """
    rng = builder.rng
    for offset, mule in enumerate(mules):
        amount = rng.uniform(4_000, 9_800) if laundering else rng.uniform(120, 9_000)
        gap = int(rng.integers(20, 240)) if laundering else int(rng.integers(40, 900))
        builder.add(int(mule), collector, amount, start + offset * gap, laundering)


def _cycle(builder: _Builder, ring: np.ndarray, start: int) -> None:
    """Round-tripping: the same value chases itself back to where it began."""
    rng = builder.rng
    amount = rng.uniform(20_000, 120_000)
    when = start
    for i in range(len(ring)):
        source, target = int(ring[i]), int(ring[(i + 1) % len(ring)])
        amount *= rng.uniform(0.94, 0.99)  # a cut taken at each hop
        when += int(rng.integers(60, 900))
        builder.add(source, target, amount, when, True)


def _chain(builder: _Builder, line: np.ndarray, start: int, laundering: bool = True) -> None:
    """Layering: value pushed forward through a line of intermediaries.

    The legitimate version is a supply chain. Same forward motion, but the
    value is not preserved hop to hop and the hops are days apart.
    """
    rng = builder.rng
    amount = rng.uniform(30_000, 200_000)
    when = start
    for i in range(len(line) - 1):
        # Laundering passes nearly the whole sum on. A supply chain does not.
        amount *= rng.uniform(0.90, 0.98) if laundering else rng.uniform(0.25, 0.75)
        when += int(rng.integers(120, 1_800)) if laundering else int(rng.integers(20_000, 90_000))
        builder.add(int(line[i]), int(line[i + 1]), amount, when, laundering)


def _decoys(builder: _Builder, accounts: int, count: int) -> None:
    """Legitimate activity shaped like laundering: payroll, settlement, supply chains."""
    rng = builder.rng
    for _ in range(count):
        kind = rng.choice(["payroll", "settlement", "supply"], p=[0.4, 0.4, 0.2])
        start = int(rng.integers(0, WINDOW_SECONDS))
        if kind == "payroll":
            members = rng.choice(accounts, size=int(rng.integers(6, 16)) + 1, replace=False)
            _fan_out(builder, int(members[0]), members[1:], start, laundering=False)
        elif kind == "settlement":
            members = rng.choice(accounts, size=int(rng.integers(6, 16)) + 1, replace=False)
            _fan_in(builder, int(members[0]), members[1:], start, laundering=False)
        else:
            _chain(builder, rng.choice(accounts, size=int(rng.integers(4, 8)), replace=False), start, False)


def synthesise(
    rows: int = 40_000,
    accounts: int = 1_200,
    patterns: int = 320,
    decoys: int = 260,
    seed: int = 0,
) -> pd.DataFrame:
    """Build the full transaction frame: background, decoys, then typologies."""
    rng = np.random.default_rng(seed)
    builder = _Builder(rng, accounts)
    _background(builder, rows)
    _decoys(builder, accounts, decoys)

    for _ in range(patterns):
        kind = rng.choice(["fan_out", "fan_in", "cycle", "chain"], p=[0.3, 0.3, 0.2, 0.2])
        start = int(rng.integers(0, WINDOW_SECONDS))
        if kind == "fan_out":
            members = rng.choice(accounts, size=int(rng.integers(6, 14)) + 1, replace=False)
            _fan_out(builder, int(members[0]), members[1:], start)
        elif kind == "fan_in":
            members = rng.choice(accounts, size=int(rng.integers(6, 14)) + 1, replace=False)
            _fan_in(builder, int(members[0]), members[1:], start)
        elif kind == "cycle":
            _cycle(builder, rng.choice(accounts, size=int(rng.integers(3, 6)), replace=False), start)
        else:
            _chain(builder, rng.choice(accounts, size=int(rng.integers(4, 8)), replace=False), start)

    frame = pd.DataFrame(builder.rows)
    return frame.sort_values("Timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=40_000, help="background (non-laundering) transactions")
    parser.add_argument("--accounts", type=int, default=1_200)
    parser.add_argument("--patterns", type=int, default=320, help="laundering typologies to plant")
    parser.add_argument(
        "--decoys", type=int, default=260, help="legitimate structures with the same shape"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = synthesise(args.rows, args.accounts, args.patterns, args.decoys, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    fraud = int(frame["Is Laundering"].sum())
    print(
        f"wrote {args.out}: {len(frame):,} transactions, "
        f"{frame['Account'].nunique():,} sending accounts, "
        f"{fraud:,} laundering ({100 * fraud / len(frame):.2f}%)"
    )


if __name__ == "__main__":
    main()
