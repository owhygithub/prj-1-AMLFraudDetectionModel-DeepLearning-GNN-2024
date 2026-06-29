#!/usr/bin/env python3
"""Generate a small fake transaction file with the IBM AML schema.

Useful for exercising the pipeline end to end without downloading the real
dataset -- the numbers it produces are meaningless, only the plumbing matters.

    python tests/make_synthetic_csv.py --rows 4000 --out data/synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CURRENCIES = ["US Dollar", "Euro", "UK Pound"]
FORMATS = ["Reinvestment", "Cheque", "Credit Card", "ACH", "Wire", "Cash"]


def synthesise(rows: int = 4000, accounts: int = 400, fraud_ratio: float = 0.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    account_ids = [f"80{i:07X}" for i in range(accounts)]
    banks = rng.integers(0, 64, size=accounts)

    source = rng.integers(0, accounts, size=rows)
    target = rng.integers(0, accounts, size=rows)
    amounts = np.round(rng.lognormal(6, 2, size=rows), 2)
    labels = (rng.random(rows) < fraud_ratio).astype(int)
    # Give the label something faint to latch onto, so training is not pure noise.
    amounts = np.where(labels == 1, np.round(amounts * 3, 2), amounts)
    currency = rng.choice(CURRENCIES, size=rows, p=[0.9, 0.07, 0.03])

    return pd.DataFrame(
        {
            "Timestamp": pd.to_datetime("2022-09-01") + pd.to_timedelta(np.sort(rng.integers(0, 86400 * 7, rows)), "s"),
            "From Bank": banks[source],
            "Account": [account_ids[i] for i in source],
            "To Bank": banks[target],
            "Account.1": [account_ids[i] for i in target],
            "Amount Received": amounts,
            "Receiving Currency": currency,
            "Amount Paid": amounts,
            "Payment Currency": currency,
            "Payment Format": rng.choice(FORMATS, size=rows),
            "Is Laundering": labels,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--accounts", type=int, default=400)
    parser.add_argument("--fraud-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = synthesise(args.rows, args.accounts, args.fraud_ratio, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(frame)} rows, {frame['Is Laundering'].sum()} laundering")


if __name__ == "__main__":
    main()
