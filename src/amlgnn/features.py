"""Feature encoders for the transaction graph.

Every encoder here is deterministic. The original code used Python's builtin
``hash()`` for account identifiers, which is salted per process, so the node
features changed on every run and nothing was reproducible.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

#: Epoch offset used to keep the integer timestamps small. The IBM AML
#: transaction files start on 2022-09-01, which is 1661990400 in Unix time.
TIMESTAMP_EPOCH = 1_661_990_400

#: Timestamps are bucketed into 10-second bins, matching the original runs.
TIMESTAMP_RESOLUTION_SECONDS = 10


def to_integer_time(timestamps: pd.Series) -> pd.Series:
    """Map a timestamp column onto small non-negative integers."""
    seconds = pd.to_datetime(timestamps).astype("int64") // 10**9
    return (seconds - TIMESTAMP_EPOCH) // TIMESTAMP_RESOLUTION_SECONDS


def digit_hash(values: pd.Series, vector_size: int = 9) -> np.ndarray:
    """Encode arbitrary strings as a fixed-length vector of decimal digits.

    Uses BLAKE2b rather than ``hash()`` so the same account always maps to the
    same vector, across processes and across machines.
    """
    modulus = 10**vector_size
    out = np.empty((len(values), vector_size), dtype=np.int64)
    for row, value in enumerate(values.astype(str)):
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        digits = str(int.from_bytes(digest, "big") % modulus).zfill(vector_size)
        out[row] = [int(d) for d in digits]
    return out


def binary_encode(values: pd.Series) -> np.ndarray:
    """Encode non-negative integers as fixed-width binary vectors."""
    ints = values.astype("int64").to_numpy()
    if ints.min() < 0:
        raise ValueError("binary_encode expects non-negative integers")
    width = max(1, int(ints.max()).bit_length())
    return ((ints[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.int64)


def first_token_dummies(values: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot encode a categorical column, keeping only its first token.

    Some rows carry comma-separated values (``"US Dollar,Euro"``); the original
    code treated the first entry as the one that actually applies.
    """
    first = values.astype(str).str.split(",", n=1).str[0].str.strip()
    return pd.get_dummies(first, prefix=prefix, dtype=np.int64)


def digit_encode_amounts(amounts: pd.Series, integer_digits: int, fraction_digits: int) -> np.ndarray:
    """Encode monetary amounts digit-by-digit at a fixed precision.

    ``1234.5`` with ``integer_digits=6`` and ``fraction_digits=2`` becomes
    ``[0, 0, 1, 2, 3, 4, 5, 0]``. This is a vectorised replacement for the
    original per-row string surgery, which was the slowest step of the build.
    """
    scale = 10**fraction_digits
    scaled = np.rint(amounts.astype(float).to_numpy() * scale).astype(np.int64)
    width = integer_digits + fraction_digits
    powers = 10 ** np.arange(width - 1, -1, -1, dtype=np.int64)
    return (scaled[:, None] // powers) % 10


def amount_layout(amounts: pd.Series, max_fraction_digits: int = 2) -> tuple[int, int]:
    """Pick how many integer / fractional digits the amount encoding needs."""
    values = amounts.astype(float)
    integer_digits = max(1, len(str(int(values.max()))))
    return integer_digits, max_fraction_digits


def minmax_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale every column to [0, 1]. Constant columns collapse to 0."""
    values = frame.astype(float)
    span = values.max() - values.min()
    scaled = (values - values.min()).div(span.replace(0, np.nan), axis=1)
    return scaled.fillna(0.0)


def time_closeness(
    frame: pd.DataFrame,
    account_column: str = "Account",
    timestamp_column: str = "Timestamp",
    gap_penalty: float = 50_000.0,
) -> pd.Series:
    """Score how soon a transaction follows the sender's previous one.

    1.0 means "immediately after the last transaction from this account";
    values near 0 mean a long gap, and an account's very first transaction is
    treated as the longest gap seen plus a penalty so that it never scores 1.
    This is the ``T`` signal in the DistMult-T / ComplEx-T variants.
    """
    integer_time = to_integer_time(frame[timestamp_column])
    order = np.lexsort((integer_time.to_numpy(), frame[account_column].to_numpy()))
    ordered_accounts = frame[account_column].to_numpy()[order]
    ordered_time = integer_time.to_numpy()[order]

    gaps = np.full(len(frame), np.nan)
    same_account = ordered_accounts[1:] == ordered_accounts[:-1]
    gaps[1:] = np.where(same_account, ordered_time[1:] - ordered_time[:-1], np.nan)

    unsorted = np.empty_like(gaps)
    unsorted[order] = gaps

    observed_max = np.nanmax(unsorted) if np.isfinite(unsorted).any() else 0.0
    first_transaction_gap = observed_max + gap_penalty
    # Adding the penalty twice keeps the widest gap strictly below 1.0 after
    # inversion, so a first-ever transaction never looks maximally "close".
    upper_bound = first_transaction_gap + gap_penalty
    filled = np.where(np.isnan(unsorted), first_transaction_gap, unsorted)

    lower_bound = np.nanmin(filled)
    span = upper_bound - lower_bound
    normalized = np.zeros_like(filled) if span == 0 else (filled - lower_bound) / span
    return pd.Series(1.0 - normalized, index=frame.index, name="time_closeness")
