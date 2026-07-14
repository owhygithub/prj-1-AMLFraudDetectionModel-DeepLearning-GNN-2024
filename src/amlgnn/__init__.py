"""Graph-based anti-money-laundering detection on the IBM synthetic AML dataset.

The pipeline turns a transaction CSV into a graph (accounts = nodes,
transactions = edges), learns node and edge embeddings with a single-layer
GNN, and scores each edge with a knowledge-graph decoder (DistMult or
ComplEx) to classify it as laundering / not laundering.
"""

__version__ = "1.0.0"

from amlgnn.models import AMLLinkScorer, GNNEncoder

__all__ = ["AMLLinkScorer", "GNNEncoder", "__version__"]
