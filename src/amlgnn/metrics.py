"""Evaluation metrics and the figures that go with them."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    mrr: float
    loss: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)

    def __str__(self) -> str:
        parts = [
            f"accuracy={self.accuracy:.4f}",
            f"precision={self.precision:.4f}",
            f"recall={self.recall:.4f}",
            f"f1={self.f1:.4f}",
            f"roc_auc={self.roc_auc:.4f}",
            f"avg_precision={self.average_precision:.4f}",
        ]
        if self.loss is not None:
            parts.insert(0, f"loss={self.loss:.4f}")
        return "  ".join(parts)


def mean_reciprocal_rank(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean of ``1 / rank`` over positives, ranking all edges by score.

    Note this is not the retrieval MRR of the knowledge-graph literature, where
    each query contributes one correct answer. Here every laundering edge in
    the evaluation set contributes a term, so the value shrinks roughly with
    the number of positives and is only comparable between runs on an
    identically sized split. ``average_precision`` is the ranking metric to
    read for absolute quality.
    """
    order = torch.argsort(scores.flatten(), descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, order.numel() + 1, dtype=torch.float32)
    positives = labels.flatten().bool()
    if not bool(positives.any()):
        return 0.0
    return float((1.0 / ranks[positives]).mean())


def evaluate(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    loss: float | None = None,
) -> ClassificationReport:
    """Score a set of edges. ``scores`` are probabilities, not logits."""
    y_score = scores.detach().cpu().numpy().ravel()
    y_true = labels.detach().cpu().numpy().ravel().astype(int)
    y_pred = (y_score >= threshold).astype(int)

    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        average_precision=float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        mrr=mean_reciprocal_rank(scores, labels),
        loss=loss,
    )


def save_figures(
    scores: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str | Path,
    threshold: float = 0.5,
    prefix: str = "test",
) -> list[Path]:
    """Write an ROC curve and a confusion matrix into ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_score = scores.detach().cpu().numpy().ravel()
    y_true = labels.detach().cpu().numpy().ravel().astype(int)
    y_pred = (y_score >= threshold).astype(int)
    written: list[Path] = []

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_path = out_dir / f"{prefix}-roc-curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    written.append(roc_path)

    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center")
    ax.set(
        xlabel="Predicted", ylabel="Actual", title="Confusion matrix",
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["clean", "laundering"], yticklabels=["clean", "laundering"],
    )
    cm_path = out_dir / f"{prefix}-confusion-matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    written.append(cm_path)

    return written
