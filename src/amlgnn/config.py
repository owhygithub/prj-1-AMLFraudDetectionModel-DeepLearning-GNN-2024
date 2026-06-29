"""Filesystem layout.

Nothing here is hard-coded to a particular machine: every location can be
overridden with an environment variable, which is what the original cluster
runs did through absolute hard-coded absolute cluster paths.

    AMLGNN_DATA_DIR      raw + prepared CSVs and cached graph artefacts
    AMLGNN_ARTIFACT_DIR  trained model checkpoints
    AMLGNN_RESULTS_DIR   run logs, figures and the aggregate runs.csv
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default)).expanduser()


DATA_DIR = _dir("AMLGNN_DATA_DIR", PROJECT_ROOT / "data")
ARTIFACT_DIR = _dir("AMLGNN_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
RESULTS_DIR = _dir("AMLGNN_RESULTS_DIR", PROJECT_ROOT / "results")

RUNS_CSV = RESULTS_DIR / "runs.csv"

#: Columns the IBM synthetic AML transaction CSV is expected to have.
REQUIRED_COLUMNS = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for path in (DATA_DIR, ARTIFACT_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
