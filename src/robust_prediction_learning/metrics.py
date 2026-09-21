from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _top_unique(
    recommended: Sequence[int] | np.ndarray,
    k: int,
) -> list[int]:
    """
    Return the first k unique recommended item IDs while preserving order.

    Ranking metrics assume that a recommender does not receive duplicate
    credit for recommending the same item multiple times.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    output: list[int] = []
    seen: set[int] = set()

    for item in recommended:
        item = int(item)

        if item in seen:
            continue

        seen.add(item)
        output.append(item)

        if len(output) >= k:
            break

    return output


def precision_at_k(
    recommended: Sequence[int] | np.ndarray,
    relevant: Iterable[int],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Precision@K =
        number of relevant recommended items in top K / K

    If fewer than K items are returned, the denominator remains K.
    """
    ranked = _top_unique(recommended, k)
    relevant_set = {int(x) for x in relevant}

    if not relevant_set:
        return 0.0

    hits = sum(item in relevant_set for item in ranked)

    return float(hits / k)


def recall_at_k(
    recommended: Sequence[int] | np.ndarray,
    relevant: Iterable[int],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Recall@K =
        number of relevant recommended items in top K
        ------------------------------------------------
        total number of relevant items
    """
    ranked = _top_unique(recommended, k)
    relevant_set = {int(x) for x in relevant}

    if not relevant_set:
        return 0.0

    hits = sum(item in relevant_set for item in ranked)

    return float(hits / len(relevant_set))


def ndcg_at_k(
    recommended: Sequence[int] | np.ndarray,
    relevant: Iterable[int],
    k: int,
) -> float:
    """
    Compute binary-relevance NDCG@K.

    DCG@K =
        sum(rel_i / log2(i + 2))

    where ranking position i is zero-indexed.

    IDCG is the DCG of the ideal ranking containing all relevant items
    as early as possible.
    """
    ranked = _top_unique(recommended, k)
    relevant_set = {int(x) for x in relevant}

    if not relevant_set:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(ranked):
        if item in relevant_set:
            dcg += 1.0 / np.log2(rank + 2.0)

    ideal_hits = min(len(relevant_set), k)

    idcg = sum(
        1.0 / np.log2(rank + 2.0)
        for rank in range(ideal_hits)
    )

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def mrr_at_k(
    recommended: Sequence[int] | np.ndarray,
    relevant: Iterable[int],
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank contribution for one user.

    Returns:
        1 / rank_of_first_relevant_item

    or 0 if no relevant item appears in the first K positions.
    """
    ranked = _top_unique(recommended, k)
    relevant_set = {int(x) for x in relevant}

    if not relevant_set:
        return 0.0

    for rank, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return float(1.0 / rank)

    return 0.0


def cumulative_loss(
    losses: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return cumulative loss over sequential observations."""
    values = np.asarray(losses, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("losses must be a one-dimensional sequence.")

    return np.cumsum(values)


def regret(
    learner_losses: Sequence[float] | np.ndarray,
    comparator_losses: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Compute cumulative regret relative to a comparator.

    R_t =
        sum_{s <= t} learner_loss_s
        -
        sum_{s <= t} comparator_loss_s

    This will become useful during the robust-learning stage.
    """
    learner = np.asarray(learner_losses, dtype=np.float64)
    comparator = np.asarray(comparator_losses, dtype=np.float64)

    if learner.shape != comparator.shape:
        raise ValueError(
            "learner_losses and comparator_losses must have equal shape."
        )

    if learner.ndim != 1:
        raise ValueError("loss sequences must be one-dimensional.")

    return np.cumsum(learner - comparator)


def aggregate_ranking_metrics(
    per_user_metrics: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """
    Compute macro averages across users.

    Each user contributes equally regardless of the number of future
    interactions they have.
    """
    if not per_user_metrics:
        return {}

    keys = per_user_metrics[0].keys()

    return {
        key: float(
            np.mean(
                [
                    metrics[key]
                    for metrics in per_user_metrics
                ]
            )
        )
        for key in keys
    }
