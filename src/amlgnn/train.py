"""Training, k-fold cross-validation and optional hyper-parameter search.

This module replaces the seven near-identical ~700-line scripts the project
started with (``1a_distmult.py`` ... ``2c_complex_tw.py``). All six reported
variants are the same code path with different flags.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from amlgnn.data import EdgeSplit, positive_weight
from amlgnn.metrics import ClassificationReport, evaluate
from amlgnn.models import AMLLinkScorer, variant_name
from amlgnn.preprocessing import TransactionGraph

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything that defines a run. Serialised verbatim into the run log."""

    decoder: str = "distmult"
    use_time: bool = False
    learn_time_weight: bool = False

    epochs: int = 100
    learning_rate: float = 0.01
    out_channels: int = 20
    weight_decay: float = 5e-4
    dropout: float = 0.1

    # Exponential learning-rate annealing, applied every `annealing_epochs`.
    annealing_rate: float = 0.01
    annealing_epochs: int = 10

    patience: int = 10
    folds: int = 5
    balance_loss: bool = True
    threshold: float = 0.5
    seed: int = 42

    @property
    def name(self) -> str:
        return variant_name(self.decoder, self.use_time, self.learn_time_weight)


@dataclass
class TrainResult:
    config: TrainConfig
    validation: ClassificationReport
    test: ClassificationReport | None
    state_dict: dict
    train_losses: list[float]
    val_losses: list[float]
    fold_reports: list[ClassificationReport]
    test_scores: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(graph: TransactionGraph, config: TrainConfig) -> AMLLinkScorer:
    return AMLLinkScorer(
        node_features=graph.data.x.size(1),
        edge_features=graph.data.edge_attr.size(1),
        out_channels=config.out_channels,
        dropout=config.dropout,
        decoder=config.decoder,
        use_time=config.use_time,
        learn_time_weight=config.learn_time_weight,
    )


def _forward(
    model: AMLLinkScorer, graph: TransactionGraph, mask: torch.Tensor
) -> torch.Tensor:
    """Logits for the edges selected by ``mask``."""
    _, _, logits = model(
        graph.data.x,
        graph.data.edge_index[:, mask],
        graph.data.edge_attr[mask],
        graph.adjacency,
        graph.time_closeness[mask] if model.use_time else None,
    )
    return logits


def _anneal(optimizer: torch.optim.Optimizer, config: TrainConfig, epoch: int) -> None:
    if config.annealing_epochs <= 0 or epoch == 0 or epoch % config.annealing_epochs:
        return
    new_lr = config.learning_rate * math.exp(-config.annealing_rate * epoch)
    for group in optimizer.param_groups:
        group["lr"] = new_lr


def train_once(
    graph: TransactionGraph,
    config: TrainConfig,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
) -> tuple[AMLLinkScorer, ClassificationReport, list[float], list[float], dict]:
    """Train on ``train_mask``, early-stop on ``val_mask``.

    Returns the model, the report at the best epoch, the two loss curves and
    the best epoch's ``state_dict``.
    """
    labels = graph.data.y
    model = build_model(graph, config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = positive_weight(labels, train_mask) if config.balance_loss else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_report: ClassificationReport | None = None
    best_state: dict = {}
    stale_epochs = 0

    for epoch in range(config.epochs):
        _anneal(optimizer, config, epoch)

        model.train()
        optimizer.zero_grad()
        logits = _forward(model, graph, train_mask)
        loss = criterion(logits, labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = _forward(model, graph, val_mask)
            val_loss = criterion(val_logits, labels[val_mask]).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_report = evaluate(
                torch.sigmoid(val_logits), labels[val_mask], config.threshold, loss=val_loss
            )
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > config.patience:
                log.debug("early stop at epoch %d", epoch)
                break

    if best_report is None:  # pragma: no cover, only reachable if epochs == 0
        raise RuntimeError("training produced no epochs; check config.epochs")

    model.load_state_dict(best_state)
    return model, best_report, train_losses, val_losses, best_state


def cross_validate(
    graph: TransactionGraph, config: TrainConfig, split: EdgeSplit
) -> TrainResult:
    """K-fold CV inside the training split. The test edges never enter a fold.

    The original scripts ran ``KFold`` over every edge in the graph and then
    reported on the held-out test mask, so each fold trained on a fifth of the
    test edges. That leak is the main reason the published numbers should be
    read as optimistic.
    """
    set_seed(config.seed)
    train_indices = split.train.nonzero(as_tuple=True)[0]
    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=config.seed)

    fold_reports: list[ClassificationReport] = []
    best_val_loss = float("inf")
    best: tuple[ClassificationReport, dict, list[float], list[float]] | None = None

    for fold, (fold_train, fold_val) in enumerate(kfold.split(train_indices), start=1):
        train_mask = torch.zeros_like(split.train)
        val_mask = torch.zeros_like(split.train)
        train_mask[train_indices[fold_train]] = True
        val_mask[train_indices[fold_val]] = True

        _, report, train_losses, val_losses, state = train_once(graph, config, train_mask, val_mask)
        fold_reports.append(report)
        log.info("fold %d/%d  %s", fold, config.folds, report)

        if report.loss is not None and report.loss < best_val_loss:
            best_val_loss = report.loss
            best = (report, state, train_losses, val_losses)

    assert best is not None
    report, state, train_losses, val_losses = best
    return TrainResult(
        config=config,
        validation=report,
        test=None,
        state_dict=state,
        train_losses=train_losses,
        val_losses=val_losses,
        fold_reports=fold_reports,
    )


def evaluate_on_test(
    graph: TransactionGraph, config: TrainConfig, result: TrainResult, split: EdgeSplit
) -> TrainResult:
    """Score the held-out test edges with the best cross-validated model."""
    model = build_model(graph, config)
    model.load_state_dict(result.state_dict)
    model.eval()

    labels = graph.data.y[split.test]
    pos_weight = positive_weight(graph.data.y, split.train) if config.balance_loss else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        logits = _forward(model, graph, split.test)
        loss = criterion(logits, labels).item()
        scores = torch.sigmoid(logits)

    report = evaluate(scores, labels, config.threshold, loss=loss)
    return replace(result, test=report, test_scores=scores, test_labels=labels)


def mean_report(reports: list[ClassificationReport]) -> dict[str, float]:
    """Per-metric mean across folds."""
    keys = [k for k, v in reports[0].as_dict().items() if v is not None]
    return {k: float(np.mean([r.as_dict()[k] for r in reports])) for k in keys}


def tune(
    graph: TransactionGraph,
    base_config: TrainConfig,
    split: EdgeSplit,
    n_trials: int = 32,
    metric: str = "average_precision",
    search_space: dict[str, list] | None = None,
) -> TrainConfig:
    """Optuna search over the grid the original runs used.

    Selection happens on the validation split only. The original tuned on
    recall, which a model can max out by flagging everything; ``f1`` or
    ``average_precision`` keeps precision honest.
    """
    import optuna

    space = search_space or {
        "epochs": [50, 100],
        "learning_rate": [0.01, 0.001, 0.0001],
        "out_channels": [10, 15, 20, 25],
        "weight_decay": [5e-4, 5e-5],
        "dropout": [0.1, 0.5],
        "annealing_rate": [0.01, 0.001],
        "annealing_epochs": [10, 20],
    }

    if base_config.decoder == "complex" and "out_channels" in space:
        # The complex decoder splits each embedding into real and imaginary
        # halves, so odd widths are not searchable.
        space = dict(space, out_channels=[c for c in space["out_channels"] if c % 2 == 0])
        if not space["out_channels"]:
            raise ValueError("no even out_channels left in the search space for the complex decoder")

    def objective(trial: "optuna.Trial") -> float:
        params = {key: trial.suggest_categorical(key, values) for key, values in space.items()}
        config = replace(base_config, **params, patience=2)
        set_seed(config.seed)
        _, report, _, _, _ = train_once(graph, config, split.train, split.val)
        return report.as_dict()[metric]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=base_config.seed)
    )
    study.optimize(objective, n_trials=n_trials)
    log.info("best %s=%.4f with %s", metric, study.best_value, study.best_params)
    return replace(base_config, **study.best_params)


def log_run(
    result: TrainResult,
    results_dir: str | Path,
    runs_csv: str | Path,
) -> tuple[Path, Path]:
    """Write a per-run text log and append a row to the aggregate CSV."""
    results_dir = Path(results_dir)
    run_dir = results_dir / "runs" / result.config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    report = result.test or result.validation
    config = asdict(result.config)

    run_path = run_dir / f"run_{timestamp}.txt"
    with run_path.open("w") as handle:
        handle.write(f"model: {result.config.name}\ntimestamp: {timestamp}\n\n[config]\n")
        for key, value in config.items():
            handle.write(f"{key}: {value}\n")
        handle.write(f"\n[{'test' if result.test else 'validation'}]\n")
        for key, value in report.as_dict().items():
            if value is not None:
                handle.write(f"{key}: {value}\n")
        if result.fold_reports:
            handle.write("\n[cross-validation means]\n")
            for key, value in mean_report(result.fold_reports).items():
                handle.write(f"{key}: {value}\n")

    runs_csv = Path(runs_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Model": result.config.name,
        "Timestamp": timestamp,
        "learning_rate": config["learning_rate"],
        "out_channels": config["out_channels"],
        "Epoch": config["epochs"],
        "Weight_decay": config["weight_decay"],
        "Dropout": config["dropout"],
        "Loss": report.loss,
        "Accuracy": report.accuracy,
        "Precision": report.precision,
        "Recall": report.recall,
        "F1 Score": report.f1,
        "MRR": report.mrr,
        "ROC-AUC": report.roc_auc,
    }
    write_header = not runs_csv.exists()
    with runs_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return run_path, runs_csv


def save_loss_curve(result: TrainResult, path: str | Path) -> Path:
    """Plot the best fold's training and validation loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(result.train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, result.train_losses, label="training loss")
    ax.plot(epochs, result.val_losses, label="validation loss")
    ax.set(xlabel="epoch", ylabel="BCE loss", title=f"{result.config.name} loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
