from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InteractionWindow:
    """
    A fixed-size chronological window of user interactions.

    Equal-interaction windows are used instead of fixed calendar windows
    so that each distribution estimate is based on approximately the same
    amount of behavioral evidence.
    """

    window_index: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    n_interactions: int
    item_counts: dict[int, int]


def _validate_nonnegative_array(
    values: np.ndarray,
    name: str,
) -> None:
    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if np.any(values < 0):
        raise ValueError(
            f"{name} cannot contain negative values."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def jensen_shannon_divergence(
    p: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Compute Jensen-Shannon divergence between two non-negative vectors.

    With base=2, the result lies in [0, 1].

    This returns the divergence itself, not the square root
    returned by scipy.spatial.distance.jensenshannon.

    Parameters
    ----------
    p, q:
        Non-negative probability or count vectors with equal shape.

    base:
        Logarithm base. Base 2 gives a convenient [0, 1] interpretation.
    """
    p_array = np.asarray(
        p,
        dtype=np.float64,
    )

    q_array = np.asarray(
        q,
        dtype=np.float64,
    )

    if p_array.shape != q_array.shape:
        raise ValueError(
            "p and q must have equal shapes."
        )

    _validate_nonnegative_array(
        p_array,
        "p",
    )

    _validate_nonnegative_array(
        q_array,
        "q",
    )

    p_sum = float(p_array.sum())
    q_sum = float(q_array.sum())

    if p_sum <= 0:
        raise ValueError(
            "p must contain positive total mass."
        )

    if q_sum <= 0:
        raise ValueError(
            "q must contain positive total mass."
        )

    if base <= 0 or base == 1:
        raise ValueError(
            "base must be positive and different from 1."
        )

    p_prob = p_array / p_sum
    q_prob = q_array / q_sum

    mixture = 0.5 * (
        p_prob + q_prob
    )

    log_base = np.log(base)

    p_mask = p_prob > 0
    q_mask = q_prob > 0

    kl_p = np.sum(
        p_prob[p_mask]
        * (
            np.log(
                p_prob[p_mask]
                / mixture[p_mask]
            )
            / log_base
        )
    )

    kl_q = np.sum(
        q_prob[q_mask]
        * (
            np.log(
                q_prob[q_mask]
                / mixture[q_mask]
            )
            / log_base
        )
    )

    js = 0.5 * (
        kl_p + kl_q
    )

    # Numerical noise can occasionally produce values
    # infinitesimally outside the theoretical interval.
    if base == 2.0:
        js = float(
            np.clip(
                js,
                0.0,
                1.0,
            )
        )

    return float(js)


def jensen_shannon_from_counts(
    counts_p: Mapping[int, int | float],
    counts_q: Mapping[int, int | float],
    base: float = 2.0,
) -> float:
    """
    Compute Jensen-Shannon divergence directly from sparse item counts.

    Only the union of observed item IDs is materialized. This is much more
    memory efficient than creating dense vectors across the entire
    40k+ artist vocabulary.
    """
    if not counts_p:
        raise ValueError(
            "counts_p cannot be empty."
        )

    if not counts_q:
        raise ValueError(
            "counts_q cannot be empty."
        )

    keys = sorted(
        set(counts_p)
        | set(counts_q)
    )

    p = np.fromiter(
        (
            float(
                counts_p.get(
                    key,
                    0.0,
                )
            )
            for key in keys
        ),
        dtype=np.float64,
    )

    q = np.fromiter(
        (
            float(
                counts_q.get(
                    key,
                    0.0,
                )
            )
            for key in keys
        ),
        dtype=np.float64,
    )

    return jensen_shannon_divergence(
        p,
        q,
        base=base,
    )


def build_user_item_counts(
    interactions: pd.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "artist_idx",
) -> dict[int, dict[int, int]]:
    """
    Build sparse interaction-count distributions for every user.

    Returns
    -------
    {
        user_id: {
            item_id: interaction_count,
            ...
        },
        ...
    }
    """
    required = {
        user_col,
        item_col,
    }

    missing = (
        required
        - set(interactions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    grouped = (
        interactions
        .groupby(
            [
                user_col,
                item_col,
            ],
            sort=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )

    profiles: dict[
        int,
        dict[int, int]
    ] = {}

    for user_idx, group in grouped.groupby(
        user_col,
        sort=False,
    ):
        profiles[int(user_idx)] = {
            int(item_idx): int(count)
            for item_idx, count
            in zip(
                group[item_col],
                group["count"],
                strict=True,
            )
        }

    return profiles


def make_equal_interaction_windows(
    interactions: pd.DataFrame,
    window_size: int,
    timestamp_col: str = "timestamp",
    item_col: str = "artist_idx",
) -> list[InteractionWindow]:
    """
    Split one user's chronological interactions into equal-size windows.

    Only complete windows are retained. A short remainder is discarded so
    that distribution estimates from different windows have equal sample
    sizes.

    Example
    -------
    780 interactions and window_size=250 produce three complete windows:

        [0:250]
        [250:500]
        [500:750]

    and the final 30 interactions are discarded.
    """
    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    required = {
        timestamp_col,
        item_col,
    }

    missing = (
        required
        - set(interactions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if interactions.empty:
        return []

    data = (
        interactions[
            [
                timestamp_col,
                item_col,
            ]
        ]
        .dropna()
        .sort_values(
            timestamp_col,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    n_complete_windows = (
        len(data)
        // window_size
    )

    windows: list[
        InteractionWindow
    ] = []

    for window_index in range(
        n_complete_windows
    ):
        start = (
            window_index
            * window_size
        )

        end = (
            start
            + window_size
        )

        window_df = data.iloc[
            start:end
        ]

        counts_series = (
            window_df[
                item_col
            ]
            .value_counts(
                sort=False
            )
        )

        item_counts = {
            int(item_idx): int(count)
            for item_idx, count
            in counts_series.items()
        }

        windows.append(
            InteractionWindow(
                window_index=window_index,
                start_timestamp=(
                    window_df[
                        timestamp_col
                    ].iloc[0]
                ),
                end_timestamp=(
                    window_df[
                        timestamp_col
                    ].iloc[-1]
                ),
                n_interactions=(
                    len(window_df)
                ),
                item_counts=item_counts,
            )
        )

    return windows


def event_mass_at_k(
    recommended: Sequence[int] | np.ndarray,
    item_counts: Mapping[int, int | float],
    k: int,
) -> float:
    """
    Fraction of observed interaction events explained by the top-K ranking.

    Unlike binary Recall@K, repeated listening is preserved.

    If a window contains:

        Artist A: 80 interactions
        Artist B: 20 interactions

    and A appears in top-K, event_mass_at_k is 0.8.

    This is useful for music because repeated listening is meaningful.
    """
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if not item_counts:
        return 0.0

    total_mass = float(
        sum(
            item_counts.values()
        )
    )

    if total_mass <= 0:
        return 0.0

    top_items: set[int] = set()

    for item in recommended[:k]:
        top_items.add(
            int(item)
        )

    covered_mass = sum(
        float(count)
        for item, count
        in item_counts.items()
        if int(item) in top_items
    )

    return float(
        covered_mass
        / total_mass
    )


def fraction_events_not_in_set(
    item_counts: Mapping[int, int | float],
    known_items: set[int],
) -> float:
    """
    Fraction of events involving items not contained in a given set.

    Used to measure user-level novelty relative to historical listening.
    """
    if not item_counts:
        return 0.0

    total = float(
        sum(
            item_counts.values()
        )
    )

    if total <= 0:
        return 0.0

    unseen = sum(
        float(count)
        for item, count
        in item_counts.items()
        if int(item)
        not in known_items
    )

    return float(
        unseen / total
    )


def fraction_events_unknown_by_mask(
    item_counts: Mapping[int, int | float],
    known_item_mask: np.ndarray,
) -> float:
    """
    Fraction of events whose items were not available to the trained model.

    This measures global item cold-start independently of behavioral shift.
    """
    if not item_counts:
        return 0.0

    total = float(
        sum(
            item_counts.values()
        )
    )

    if total <= 0:
        return 0.0

    unknown = 0.0

    for item, count in item_counts.items():
        item = int(item)

        is_known = (
            0 <= item
            < len(known_item_mask)
            and bool(
                known_item_mask[item]
            )
        )

        if not is_known:
            unknown += float(count)

    return float(
        unknown / total
    )


def filter_counts_by_known_mask(
    item_counts: Mapping[int, int | float],
    known_item_mask: np.ndarray,
) -> dict[int, int]:
    """
    Keep only items representable by a fitted historical model.
    """
    output: dict[int, int] = {}

    for item, count in item_counts.items():
        item = int(item)

        if (
            0 <= item
            < len(known_item_mask)
            and bool(
                known_item_mask[item]
            )
        ):
            output[item] = int(count)

    return output
