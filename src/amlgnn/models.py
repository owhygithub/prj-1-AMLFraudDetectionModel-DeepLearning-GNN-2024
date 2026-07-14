"""The GNN encoder and the knowledge-graph decoders that score transactions.

Architecture (one layer, as in ``docs/images/architecture.png``)::

    Zx = dropout( (A @ X) @ W_node )      node embeddings
    Ze = dropout(  E     @ W_edge )       edge embeddings
    score(head, relation, tail) -> logit  DistMult or ComplEx

A transaction is an edge (head account -> tail account) whose own features act
as the relation, so classifying a transaction is exactly the triple-scoring
task that DistMult and ComplEx were designed for.

The decoders return **logits**, not probabilities. The original scripts applied
``torch.sigmoid`` inside the decoder and then fed the result to
``BCEWithLogitsLoss``, which sigmoids again -- see ``RESULTS.md`` for what that
cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DECODERS = ("distmult", "complex")


class GNNEncoder(nn.Module):
    """One propagation step over the account graph plus an edge projection.

    Parameters
    ----------
    node_features, edge_features:
        Input dimensions of ``x`` and ``edge_attr``.
    out_channels:
        Embedding width. For the ``complex`` decoder this must be even: the
        first half of each embedding is the real part, the second the imaginary
        part.
    dropout:
        Dropout applied to both embeddings.
    """

    def __init__(self, node_features: int, edge_features: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.weight_node = nn.Parameter(torch.empty(node_features, out_channels))
        self.weight_edge = nn.Parameter(torch.empty(edge_features, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_node)
        nn.init.xavier_uniform_(self.weight_edge)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_embeddings, edge_embeddings)``.

        ``adjacency`` is passed in explicitly; the original layer reached for a
        module-level global, which made the model impossible to reuse or test.
        """
        aggregated = torch.sparse.mm(adjacency, x) if adjacency.is_sparse else adjacency @ x
        node_embedding = self.dropout(aggregated @ self.weight_node)
        edge_embedding = self.dropout(edge_attr @ self.weight_edge)
        return node_embedding, edge_embedding


def distmult_score(head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """<h, r, t> -- the sum of the element-wise product of the three vectors."""
    return (head * relation * tail).sum(dim=-1)


def complex_score(head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Re(<h, r, conj(t)>) over embeddings split into real / imaginary halves.

    The original code built complex tensors with ``torch.zeros_like`` as the
    imaginary part, so ``conj`` was a no-op and this reduced exactly to
    DistMult. Splitting the embedding in half gives the model a genuine
    imaginary component and therefore the asymmetry ComplEx is chosen for --
    which matters here because "A paid B" is not the same event as "B paid A".
    """
    head_re, head_im = head.chunk(2, dim=-1)
    rel_re, rel_im = relation.chunk(2, dim=-1)
    tail_re, tail_im = tail.chunk(2, dim=-1)
    return (
        (head_re * rel_re * tail_re).sum(dim=-1)
        + (head_re * rel_im * tail_im).sum(dim=-1)
        + (head_im * rel_re * tail_im).sum(dim=-1)
        - (head_im * rel_im * tail_re).sum(dim=-1)
    )


class AMLLinkScorer(nn.Module):
    """Encoder + decoder, optionally gated by how soon a transaction follows.

    Variants, matching the names used in ``results/runs.csv``:

    ======================  ==============  ====================
    Variant                 ``use_time``    ``learn_time_weight``
    ======================  ==============  ====================
    ``DistMult`` / ``ComplEx``   False            False
    ``-T``                       True             False
    ``-T+W``                     True             True
    ======================  ==============  ====================
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        out_channels: int,
        dropout: float = 0.1,
        decoder: str = "distmult",
        use_time: bool = False,
        learn_time_weight: bool = False,
    ):
        super().__init__()
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")
        if decoder == "complex" and out_channels % 2:
            raise ValueError("the complex decoder needs an even out_channels")
        if learn_time_weight and not use_time:
            raise ValueError("learn_time_weight requires use_time=True")

        self.decoder = decoder
        self.use_time = use_time
        self.encoder = GNNEncoder(node_features, edge_features, out_channels, dropout)
        # Held as a plain parameter and used as-is. The original wrapped it in
        # torch.tensor(...) at every forward pass, which detaches it from the
        # autograd graph, so the "learned" weight never moved off its init.
        self.time_weight = nn.Parameter(torch.empty(1)) if learn_time_weight else None
        self.reset_parameters()

    @property
    def learn_time_weight(self) -> bool:
        return self.time_weight is not None

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        if self.time_weight is not None:
            nn.init.normal_(self.time_weight, mean=1.0, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        adjacency: torch.Tensor,
        time_closeness: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(node_embeddings, edge_embeddings, logits)``."""
        node_embedding, edge_embedding = self.encoder(x, edge_attr, adjacency)
        head = node_embedding[edge_index[0]]
        tail = node_embedding[edge_index[1]]

        score_fn = distmult_score if self.decoder == "distmult" else complex_score
        logits = score_fn(head, edge_embedding, tail)

        if self.use_time:
            if time_closeness is None:
                raise ValueError("this model was built with use_time=True but got no time_closeness")
            gate = time_closeness
            if self.time_weight is not None:
                gate = gate * self.time_weight
            logits = logits * gate

        return node_embedding, edge_embedding, logits

    @torch.no_grad()
    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Laundering probability per edge."""
        return torch.sigmoid(self(*args, **kwargs)[2])


def variant_name(decoder: str, use_time: bool, learn_time_weight: bool) -> str:
    """Human-readable variant label, e.g. ``ComplEx-T+W``."""
    base = {"distmult": "DistMult", "complex": "ComplEx"}[decoder]
    if learn_time_weight:
        return f"{base}-T+W"
    if use_time:
        return f"{base}-T"
    return base
