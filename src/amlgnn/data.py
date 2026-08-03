"""Train / validation / test splits over transaction edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class EdgeSplit:
    """Boolean masks over the edges of a :class:`~amlgnn.preprocessing.TransactionGraph`."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor

    def sizes(self) -> dict[str, int]:
        return {name: int(getattr(self, name).sum()) for name in ("train", "val", "test")}

    def __post_init__(self) -> None:
        overlap = (self.train & self.val) | (self.train & self.test) | (self.val & self.test)
        if bool(overlap.any()):
            raise ValueError("train/val/test edge masks overlap")


def split_edges(
    labels: torch.Tensor,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> EdgeSplit:
    """Stratified split of edge indices into train / validation / test."""
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be in (0, 1)")

    num_edges = labels.numel()
    y = labels.cpu().numpy()
    holdout = val_fraction + test_fraction

    train_idx, rest_idx = train_test_split(
        range(num_edges), test_size=holdout, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_fraction / holdout, stratify=y[rest_idx], random_state=seed
    )

    def mask(indices) -> torch.Tensor:
        out = torch.zeros(num_edges, dtype=torch.bool)
        out[torch.as_tensor(list(indices), dtype=torch.long)] = True
        return out

    return EdgeSplit(train=mask(train_idx), val=mask(val_idx), test=mask(test_idx))


def positive_weight(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``negatives / positives`` for ``BCEWithLogitsLoss(pos_weight=...)``.

    The balanced dataset used for the reported runs is close to 50/50 so this
    is near 1, but on the raw IBM files laundering is well under 1% of rows and
    an unweighted loss collapses to predicting "clean" for everything.
    """
    subset = labels[mask]
    positives = subset.sum()
    if positives == 0:
        return torch.tensor(1.0)
    return (subset.numel() - positives) / positives
