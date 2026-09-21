from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_prediction_learning.data import (
    preprocess_interactions,
    temporal_split,
    validate_temporal_split,
)
from robust_prediction_learning.metrics import (
    cumulative_loss,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    regret,
)
from robust_prediction_learning.recommenders import (
    BPRRecommender,
    PopularityRecommender,
)


# ============================================================
# DATA TESTS
# ============================================================


def make_toy_dataset() -> pd.DataFrame:
    rows = []

    for user in ["u1", "u2"]:
        for i in range(20):
            rows.append(
                {
                    "user_id": user,
                    "timestamp": (
                        f"2026-01-{i + 1:02d}"
                        "T12:00:00Z"
                    ),
                    "artist_id": f"a{i % 4}",
                    "artist_name": (
                        f"Artist {i % 4}"
                    ),
                    "track_id": f"t{i}",
                    "track_name": f"Track {i}",
                }
            )

    return pd.DataFrame(rows)


def test_preprocessing_sorts_interactions():
    df = make_toy_dataset()

    processed = preprocess_interactions(
        df,
        min_user_interactions=1,
        min_artist_interactions=1,
    )

    for _, group in processed.groupby(
        "user_idx"
    ):
        timestamps = group[
            "timestamp"
        ].to_numpy()

        assert np.all(
            timestamps[:-1]
            <= timestamps[1:]
        )


def test_temporal_split_sizes():
    df = make_toy_dataset()

    processed = preprocess_interactions(
        df,
        min_user_interactions=1,
        min_artist_interactions=1,
    )

    split = temporal_split(
        processed,
        train_fraction=0.70,
        validation_fraction=0.10,
    )

    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert len(split.test) > 0


def test_temporal_split_has_no_leakage():
    df = make_toy_dataset()

    processed = preprocess_interactions(
        df,
        min_user_interactions=1,
        min_artist_interactions=1,
    )

    split = temporal_split(processed)

    validate_temporal_split(split)


def test_user_and_artist_indices_are_integer():
    df = make_toy_dataset()

    processed = preprocess_interactions(
        df,
        min_user_interactions=1,
        min_artist_interactions=1,
    )

    assert pd.api.types.is_integer_dtype(
        processed["user_idx"]
    )

    assert pd.api.types.is_integer_dtype(
        processed["artist_idx"]
    )


# ============================================================
# RANKING METRIC TESTS
# ============================================================


def test_recall_at_k():
    recommended = [2, 8, 5]
    relevant = {2, 5}

    result = recall_at_k(
        recommended,
        relevant,
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_at_k():
    recommended = [2, 8, 5]
    relevant = {2, 5}

    result = ndcg_at_k(
        recommended,
        relevant,
        k=3,
    )

    expected_dcg = (
        1.0
        + 1.0 / np.log2(4.0)
    )

    expected_idcg = (
        1.0
        + 1.0 / np.log2(3.0)
    )

    expected = (
        expected_dcg
        / expected_idcg
    )

    assert result == pytest.approx(
        expected
    )


def test_mrr_at_k():
    recommended = [9, 7, 5, 2]
    relevant = {5}

    result = mrr_at_k(
        recommended,
        relevant,
        k=4,
    )

    assert result == pytest.approx(
        1.0 / 3.0
    )


def test_empty_relevant_set_returns_zero():
    recommended = [1, 2, 3]

    assert recall_at_k(
        recommended,
        set(),
        k=3,
    ) == 0.0

    assert ndcg_at_k(
        recommended,
        set(),
        k=3,
    ) == 0.0

    assert mrr_at_k(
        recommended,
        set(),
        k=3,
    ) == 0.0


# ============================================================
# ONLINE METRIC TESTS
# ============================================================


def test_cumulative_loss():
    losses = [1.0, 0.5, 0.25]

    result = cumulative_loss(losses)

    expected = np.array(
        [1.0, 1.5, 1.75]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_regret():
    learner = [1.0, 0.8, 0.4]
    comparator = [0.7, 0.5, 0.3]

    result = regret(
        learner,
        comparator,
    )

    expected = np.array(
        [0.3, 0.6, 0.7]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


# ============================================================
# RECOMMENDER TESTS
# ============================================================


def make_recommender_interactions() -> pd.DataFrame:
    """
    Create a small implicit-feedback matrix.

    user 0: items 0, 1
    user 1: items 1, 2
    user 2: items 2, 3
    user 3: items 3, 4
    user 4: items 1, 4
    """
    return pd.DataFrame(
        {
            "user_idx": [
                0, 0,
                1, 1,
                2, 2,
                3, 3,
                4, 4,
            ],
            "artist_idx": [
                0, 1,
                1, 2,
                2, 3,
                3, 4,
                1, 4,
            ],
        }
    )


def test_popularity_recommender_excludes_seen_items():
    interactions = (
        make_recommender_interactions()
    )

    model = PopularityRecommender()
    model.fit(interactions)

    recommendations = model.recommend(
        user_idx=0,
        k=3,
        exclude_seen=True,
    )

    assert 0 not in recommendations
    assert 1 not in recommendations

    assert len(recommendations) == 3


def test_popularity_prefers_more_common_item():
    interactions = (
        make_recommender_interactions()
    )

    model = PopularityRecommender()
    model.fit(interactions)

    # User 2 has seen items 2 and 3.
    # Item 1 is the most popular unseen item.
    recommendations = model.recommend(
        user_idx=2,
        k=1,
        exclude_seen=True,
    )

    assert recommendations[0] == 1


def test_bpr_recommender_fits_and_scores():
    interactions = (
        make_recommender_interactions()
    )

    model = BPRRecommender(
        n_factors=8,
        learning_rate=0.05,
        regularization=0.001,
        epochs=3,
        batch_size=4,
        max_samples_per_epoch=None,
        seed=42,
        verbose=False,
    )

    model.fit(interactions)

    scores = model.score_all_items(
        user_idx=0,
        exclude_seen=False,
    )

    assert scores.shape == (
        model.n_items_,
    )

    known_scores = scores[
        model.known_item_mask_
    ]

    assert np.all(
        np.isfinite(known_scores)
    )


def test_bpr_recommender_excludes_seen_items():
    interactions = (
        make_recommender_interactions()
    )

    model = BPRRecommender(
        n_factors=8,
        epochs=2,
        batch_size=4,
        max_samples_per_epoch=None,
        seed=123,
        verbose=False,
    )

    model.fit(interactions)

    recommendations = model.recommend(
        user_idx=0,
        k=3,
        exclude_seen=True,
    )

    assert 0 not in recommendations
    assert 1 not in recommendations

    assert len(recommendations) == 3


from robust_prediction_learning.shift import (
    event_mass_at_k,
    jensen_shannon_divergence,
    jensen_shannon_from_counts,
    make_equal_interaction_windows,
)


def test_js_identical_distributions_is_zero():
    p = [0.2, 0.3, 0.5]

    assert jensen_shannon_divergence(
        p,
        p,
    ) == pytest.approx(0.0)


def test_js_is_symmetric():
    p = [0.8, 0.2]
    q = [0.1, 0.9]

    assert jensen_shannon_divergence(
        p,
        q,
    ) == pytest.approx(
        jensen_shannon_divergence(
            q,
            p,
        )
    )


def test_js_disjoint_distributions_is_one():
    p = [1.0, 0.0]
    q = [0.0, 1.0]

    assert jensen_shannon_divergence(
        p,
        q,
        base=2.0,
    ) == pytest.approx(1.0)


def test_sparse_js_matches_dense_js():
    sparse = jensen_shannon_from_counts(
        {1: 3, 2: 1},
        {2: 1, 3: 3},
    )

    dense = jensen_shannon_divergence(
        [3, 1, 0],
        [0, 1, 3],
    )

    assert sparse == pytest.approx(
        dense
    )


def test_event_mass_at_k():
    counts = {
        1: 8,
        2: 2,
    }

    recommendations = [1, 3]

    assert event_mass_at_k(
        recommendations,
        counts,
        k=2,
    ) == pytest.approx(0.8)
