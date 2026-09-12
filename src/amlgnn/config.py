"""Filesystem layout.

Nothing here is hard-coded to a particular machine: every location can be
overridden with an environment variable, which is what the original cluster
runs did through absolute hard-coded absolute cluster paths.

    AMLGNN_DATA_DIR      raw + prepared CSVs and cached graph artefacts
    AMLGNN_ARTIFACT_DIR  trained model checkpoints
    AMLGNN_OUTPUT_DIR    run logs, figures and the aggregate runs.csv

New runs write to ``output/``, not to ``results/``. ``results/`` holds the
committed record of the 2024 experiments that RESULTS.md reports on, and
nothing should overwrite it.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default)).expanduser()


DATA_DIR = _dir("AMLGNN_DATA_DIR", PROJECT_ROOT / "data")
ARTIFACT_DIR = _dir("AMLGNN_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
OUTPUT_DIR = _dir("AMLGNN_OUTPUT_DIR", PROJECT_ROOT / "output")

RUNS_CSV = OUTPUT_DIR / "runs.csv"

#: The committed 2024 experiment record. Read only. See RESULTS.md.
ARCHIVED_RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVED_RUNS_CSV = ARCHIVED_RESULTS_DIR / "runs.csv"

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
    for path in (DATA_DIR, ARTIFACT_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
