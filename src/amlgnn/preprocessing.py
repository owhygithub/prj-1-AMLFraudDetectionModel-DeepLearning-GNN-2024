"""Turn a transaction CSV into the graph the model trains on.

Accounts become nodes, transactions become edges, and ``Is Laundering``
becomes the per-edge label. Node features describe the account (bank,
currency, identity hash); edge features describe the transaction (amount,
currency, payment format, time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

from amlgnn import features as feat
from amlgnn.config import REQUIRED_COLUMNS

log = logging.getLogger(__name__)


def _sparse(indices: torch.Tensor, values: torch.Tensor, shape) -> torch.Tensor:
    """Build a coalesced sparse COO tensor from indices we know are valid."""
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, values, tuple(shape), dtype=torch.float32).coalesce()


@dataclass
class GraphData:
    """The tensors that describe the graph.

    A stand-in for ``torch_geometric.data.Data``. The original code depended on
    PyTorch Geometric for this container and for a ``MessagePassing`` base
    class whose ``propagate`` was never called -- the layer is a plain sparse
    matmul. Dropping the dependency also drops ``torch_scatter`` and the
    hand-built ``torch_sparse==0.6.9`` wheel that used to be vendored in the
    repository, and makes ``pip install -r requirements.txt`` just work.
    """

    x: torch.Tensor
    """``[num_nodes, num_node_features]`` account features."""

    edge_index: torch.Tensor
    """``[2, num_edges]`` sender / receiver node indices per transaction."""

    edge_attr: torch.Tensor
    """``[num_edges, num_edge_features]`` transaction features."""

    y: torch.Tensor
    """``[num_edges]`` 1.0 where ``Is Laundering`` is set."""

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    def __repr__(self) -> str:
        return (
            f"GraphData(x={list(self.x.shape)}, edge_index={list(self.edge_index.shape)}, "
            f"edge_attr={list(self.edge_attr.shape)}, y={list(self.y.shape)})"
        )


@dataclass
class TransactionGraph:
    """Everything the training scripts need, in one serialisable object."""

    data: GraphData
    """Node features, edge index, edge features and labels."""

    adjacency: torch.Tensor
    """Sparse COO account-by-account adjacency used for message passing."""

    time_closeness: torch.Tensor
    """Per-edge recency signal in [0, 1]; see :func:`amlgnn.features.time_closeness`."""

    accounts: pd.Series
    """Account identifier for each node index."""

    node_feature_names: list[str]
    edge_feature_names: list[str]

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_edges(self) -> int:
        return self.data.num_edges

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        adjacency = self.adjacency.coalesce()
        torch.save(
            {
                "data": self.data,
                # Stored as plain tensors rather than a sparse tensor: torch
                # rebuilds those on load with an invariant check that emits a
                # warning on every run.
                "adjacency_indices": adjacency.indices(),
                "adjacency_values": adjacency.values(),
                "adjacency_shape": tuple(adjacency.shape),
                "time_closeness": self.time_closeness,
                "accounts": self.accounts,
                "node_feature_names": self.node_feature_names,
                "edge_feature_names": self.edge_feature_names,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TransactionGraph":
        payload = torch.load(Path(path), weights_only=False)
        adjacency = _sparse(
            payload.pop("adjacency_indices"),
            payload.pop("adjacency_values"),
            payload.pop("adjacency_shape"),
        )
        return cls(adjacency=adjacency, **payload)


def _node_table(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per account, carrying the bank and currency first seen for it."""
    both_sides = pd.DataFrame(
        {
            "Account": pd.concat([transactions["Account"], transactions["Account.1"]], ignore_index=True),
            "Bank": pd.concat([transactions["From Bank"], transactions["To Bank"]], ignore_index=True),
            "Currency": pd.concat(
                [transactions["Receiving Currency"], transactions["Payment Currency"]], ignore_index=True
            ),
        }
    )
    return both_sides.drop_duplicates(subset="Account").reset_index(drop=True)


def build_node_features(nodes: pd.DataFrame) -> pd.DataFrame:
    """Currency one-hot + normalised account hash + binary bank id."""
    currency = feat.first_token_dummies(nodes["Currency"], prefix="currency")
    account_hash = pd.DataFrame(
        feat.digit_hash(nodes["Account"]),
        columns=[f"account_{i}" for i in range(9)],
    )
    bank_bits = feat.binary_encode(nodes["Bank"])
    bank = pd.DataFrame(bank_bits, columns=[f"bank_{i}" for i in range(bank_bits.shape[1])])
    frame = pd.concat(
        [currency.reset_index(drop=True), feat.minmax_normalize(account_hash), bank],
        axis=1,
    )
    # The original inserted a column of random floats as a per-node "unique
    # ID". It carried no information and made every run differ; dropped.
    return frame


def build_edge_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Normalised timestamp + digit-encoded amount + currency / format one-hots."""
    integer_digits, fraction_digits = feat.amount_layout(transactions["Amount Paid"])
    amount_digits = feat.digit_encode_amounts(transactions["Amount Paid"], integer_digits, fraction_digits)
    amount = pd.DataFrame(
        amount_digits,
        columns=[f"amount_{i}" for i in range(amount_digits.shape[1])],
    )
    timestamp = feat.to_integer_time(transactions["Timestamp"]).to_frame("timestamp")
    scaled = feat.minmax_normalize(pd.concat([timestamp.reset_index(drop=True), amount], axis=1))
    categorical = pd.concat(
        [
            feat.first_token_dummies(transactions["Payment Currency"], prefix="paid_in"),
            feat.first_token_dummies(transactions["Payment Format"], prefix="format"),
        ],
        axis=1,
    ).reset_index(drop=True)
    return pd.concat([scaled, categorical], axis=1)


def _sparse_adjacency(num_nodes: int, sources: np.ndarray, targets: np.ndarray) -> torch.Tensor:
    """Binary, symmetric adjacency with parallel edges collapsed to one.

    The original build went through a dense ``networkx`` matrix, which is
    O(accounts^2) in memory -- 100k accounts alone would need 40 GB. Building
    the sparse matrix directly is what makes the large dataset tractable.
    """
    both = np.concatenate([sources, targets]), np.concatenate([targets, sources])
    values = np.ones(len(both[0]), dtype=np.float32)
    matrix = coo_matrix((values, both), shape=(num_nodes, num_nodes))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col])).long()
    return _sparse(indices, torch.from_numpy(matrix.data), matrix.shape)


def build_graph(transactions: pd.DataFrame) -> TransactionGraph:
    """Build the full :class:`TransactionGraph` from a transaction frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"transaction frame is missing columns: {missing}")

    transactions = transactions.reset_index(drop=True)
    log.info("building graph from %d transactions", len(transactions))

    nodes = _node_table(transactions)
    index_of = pd.Series(nodes.index, index=nodes["Account"])
    sources = transactions["Account"].map(index_of).to_numpy()
    targets = transactions["Account.1"].map(index_of).to_numpy()

    node_features = build_node_features(nodes)
    edge_features = build_edge_features(transactions)
    log.info(
        "%d accounts x %d node features, %d transactions x %d edge features",
        len(node_features),
        node_features.shape[1],
        len(edge_features),
        edge_features.shape[1],
    )

    data = GraphData(
        x=torch.tensor(node_features.to_numpy(dtype=np.float32)),
        edge_index=torch.tensor(np.vstack([sources, targets]), dtype=torch.long),
        edge_attr=torch.tensor(edge_features.to_numpy(dtype=np.float32)),
        y=torch.tensor(transactions["Is Laundering"].to_numpy(dtype=np.float32)),
    )

    return TransactionGraph(
        data=data,
        adjacency=_sparse_adjacency(len(node_features), sources, targets),
        time_closeness=torch.tensor(feat.time_closeness(transactions).to_numpy(dtype=np.float32)),
        accounts=nodes["Account"],
        node_feature_names=list(node_features.columns),
        edge_feature_names=list(edge_features.columns),
    )


def load_transactions(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a transaction CSV, keeping ``Timestamp`` as a datetime."""
    return pd.read_csv(Path(path), nrows=nrows, parse_dates=["Timestamp"])


def networkx_graph(transactions: pd.DataFrame, limit: int | None = None) -> nx.DiGraph:
    """A small ``networkx`` view of the transactions, for plotting only."""
    frame = transactions if limit is None else transactions.head(limit)
    graph = nx.DiGraph()
    for source, target, amount, label in zip(
        frame["Account"], frame["Account.1"], frame["Amount Paid"], frame["Is Laundering"]
    ):
        graph.add_edge(source, target, amount=float(amount), is_laundering=int(label))
    return graph
