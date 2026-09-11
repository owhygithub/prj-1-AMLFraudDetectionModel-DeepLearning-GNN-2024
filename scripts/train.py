#!/usr/bin/env python3
"""Train one model variant on a cached graph and write its results.

    # DistMult, no time signal
    python scripts/train.py --graph data/graph.pt --decoder distmult

    # ComplEx with a learned time weight ("ComplEx-T+W"), tuned first
    python scripts/train.py --graph data/graph.pt --decoder complex \
        --use-time --learn-time-weight --tune --trials 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from amlgnn.cli import setup_logging
from amlgnn.config import ARTIFACT_DIR, DATA_DIR, OUTPUT_DIR, RUNS_CSV, ensure_dirs
from amlgnn.data import split_edges
from amlgnn.metrics import save_figures
from amlgnn.preprocessing import TransactionGraph
from amlgnn.train import (
    TrainConfig,
    cross_validate,
    evaluate_on_test,
    log_run,
    mean_report,
    save_loss_curve,
    tune,
)

log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, default=DATA_DIR / "graph.pt")

    variant = parser.add_argument_group("variant")
    variant.add_argument("--decoder", choices=("distmult", "complex"), default="distmult")
    variant.add_argument("--use-time", action="store_true", help="gate scores by transaction recency (-T)")
    variant.add_argument(
        "--learn-time-weight", action="store_true", help="learn the recency gate's scale (-T+W)"
    )

    hyper = parser.add_argument_group("hyper-parameters")
    hyper.add_argument("--epochs", type=int, default=100)
    hyper.add_argument("--learning-rate", type=float, default=0.01)
    hyper.add_argument("--out-channels", type=int, default=20)
    hyper.add_argument("--weight-decay", type=float, default=5e-4)
    hyper.add_argument("--dropout", type=float, default=0.1)
    hyper.add_argument("--folds", type=int, default=5)
    hyper.add_argument("--patience", type=int, default=10)
    hyper.add_argument("--threshold", type=float, default=0.5)
    hyper.add_argument(
        "--no-balance-loss", action="store_true", help="disable the pos_weight class re-weighting"
    )

    search = parser.add_argument_group("hyper-parameter search")
    search.add_argument("--tune", action="store_true", help="run an Optuna search before training")
    search.add_argument("--trials", type=int, default=32)
    search.add_argument(
        "--tune-metric",
        default="average_precision",
        choices=("average_precision", "f1", "recall", "precision", "roc_auc"),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="where run logs and figures go; results/ is the committed 2024 record",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    ensure_dirs()

    if args.learn_time_weight and not args.use_time:
        log.info("--learn-time-weight implies --use-time")
        args.use_time = True

    graph = TransactionGraph.load(args.graph)
    log.info("loaded %s: %d accounts, %d transactions", args.graph, graph.num_nodes, graph.num_edges)

    config = TrainConfig(
        decoder=args.decoder,
        use_time=args.use_time,
        learn_time_weight=args.learn_time_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        out_channels=args.out_channels,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        folds=args.folds,
        patience=args.patience,
        threshold=args.threshold,
        balance_loss=not args.no_balance_loss,
        seed=args.seed,
    )
    split = split_edges(graph.data.y, seed=args.seed)
    log.info("variant %s, split %s", config.name, split.sizes())

    if args.tune:
        log.info("tuning %d trials on %s", args.trials, args.tune_metric)
        config = tune(graph, config, split, n_trials=args.trials, metric=args.tune_metric)
        log.info("tuned config: %s", config)

    result = cross_validate(graph, config, split)
    log.info("cross-validation means: %s", {k: round(v, 4) for k, v in mean_report(result.fold_reports).items()})

    result = evaluate_on_test(graph, config, result, split)
    log.info("TEST  %s", result.test)

    checkpoint = Path(args.artifact_dir) / f"{config.name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.state_dict,
            "config": config,
            "node_features": graph.data.x.size(1),
            "edge_features": graph.data.edge_attr.size(1),
        },
        checkpoint,
    )
    log.info("saved checkpoint %s", checkpoint)

    run_path, runs_csv = log_run(result, args.output_dir, Path(args.output_dir) / RUNS_CSV.name)
    log.info("logged %s and appended to %s", run_path, runs_csv)

    if not args.no_figures and result.test_scores is not None:
        figure_dir = Path(args.output_dir) / "figures" / config.name
        for path in save_figures(result.test_scores, result.test_labels, figure_dir):
            log.info("wrote %s", path)
        log.info("wrote %s", save_loss_curve(result, figure_dir / "validation-losses.png"))


if __name__ == "__main__":
    main()
