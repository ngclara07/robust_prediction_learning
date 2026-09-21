from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit


REQUIRED_COLUMNS = {"user_idx", "artist_idx"}


def _validate_interactions(interactions: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(interactions.columns)

    if missing:
        raise ValueError(
            f"Interactions are missing required columns: {missing}"
        )

    if interactions.empty:
        raise ValueError("Interaction dataframe is empty.")


def _top_k_from_scores(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Efficiently return the indices of the K highest finite scores.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    scores = np.asarray(scores)

    finite_items = np.flatnonzero(np.isfinite(scores))

    if len(finite_items) == 0:
        return np.empty(0, dtype=np.int64)

    k = min(k, len(finite_items))

    if len(finite_items) <= k:
        order = np.argsort(
            scores[finite_items]
        )[::-1]

        return finite_items[order].astype(np.int64)

    finite_scores = scores[finite_items]

    candidate_positions = np.argpartition(
        finite_scores,
        -k,
    )[-k:]

    candidates = finite_items[candidate_positions]

    order = np.argsort(
        scores[candidates]
    )[::-1]

    return candidates[order].astype(np.int64)


class PopularityRecommender:
    """
    Global-popularity recommendation baseline.

    Popularity is measured using the number of distinct training users
    who interacted with each artist.

    This avoids allowing a single user's repeated listening to dominate
    the popularity score.
    """

    def __init__(self) -> None:
        self.n_users_: int | None = None
        self.n_items_: int | None = None

        self.item_scores_: np.ndarray | None = None

        self.known_user_mask_: np.ndarray | None = None
        self.known_item_mask_: np.ndarray | None = None
        self.seen_matrix_: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
    ) -> "PopularityRecommender":
        _validate_interactions(interactions)

        pairs = (
            interactions[
                ["user_idx", "artist_idx"]
            ]
            .drop_duplicates()
            .astype(
                {
                    "user_idx": "int64",
                    "artist_idx": "int64",
                }
            )
        )

        users = pairs["user_idx"].to_numpy()
        items = pairs["artist_idx"].to_numpy()

        self.n_users_ = int(users.max()) + 1
        self.n_items_ = int(items.max()) + 1

        self.known_user_mask_ = np.zeros(
            self.n_users_,
            dtype=bool,
        )

        self.known_item_mask_ = np.zeros(
            self.n_items_,
            dtype=bool,
        )

        self.known_user_mask_[np.unique(users)] = True
        self.known_item_mask_[np.unique(items)] = True

        # ~40 MB for the present Last.fm dimensions, which is acceptable
        # and makes exclusion and BPR negative sampling efficient.
        self.seen_matrix_ = np.zeros(
            (self.n_users_, self.n_items_),
            dtype=bool,
        )

        self.seen_matrix_[users, items] = True

        # Since pairs are user-item unique, bincount measures the number
        # of distinct users who have interacted with each item.
        self.item_scores_ = np.bincount(
            items,
            minlength=self.n_items_,
        ).astype(np.float64)

        return self

    def score_all_items(
        self,
        user_idx: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        if self.item_scores_ is None:
            raise RuntimeError("Model has not been fitted.")

        scores = self.item_scores_.copy()

        scores[
            ~self.known_item_mask_
        ] = -np.inf

        if (
            exclude_seen
            and 0 <= user_idx < self.n_users_
            and self.known_user_mask_[user_idx]
        ):
            scores[
                self.seen_matrix_[user_idx]
            ] = -np.inf

        return scores

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        scores = self.score_all_items(
            user_idx=user_idx,
            exclude_seen=exclude_seen,
        )

        return _top_k_from_scores(scores, k)


class BPRRecommender:
    """
    Bayesian Personalized Ranking matrix-factorization recommender.

    The implementation uses mini-batch stochastic optimization over
    implicit user-item interactions.

    Objective for one triplet (u, i, j):

        -log sigmoid(
            b_i - b_j
            + p_u^T q_i
            - p_u^T q_j
        )

    where:
        i = observed positive item
        j = sampled unobserved item

    Notes
    -----
    The implementation deliberately remains compact and reproducible
    rather than attempting to match highly optimized recommender
    libraries. It is sufficient for establishing the historical ML
    predictor used by the research project.
    """

    def __init__(
        self,
        n_factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.005,
        epochs: int = 5,
        batch_size: int = 8192,
        max_samples_per_epoch: int | None = 500_000,
        init_std: float = 0.01,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if n_factors <= 0:
            raise ValueError("n_factors must be positive.")

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if regularization < 0:
            raise ValueError(
                "regularization cannot be negative."
            )

        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_samples_per_epoch = max_samples_per_epoch
        self.init_std = init_std
        self.seed = seed
        self.verbose = verbose

        self.n_users_: int | None = None
        self.n_items_: int | None = None

        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.item_bias_: np.ndarray | None = None

        self.known_user_mask_: np.ndarray | None = None
        self.known_item_mask_: np.ndarray | None = None
        self.known_item_ids_: np.ndarray | None = None

        self.seen_matrix_: np.ndarray | None = None

        self.training_history_: list[dict[str, float]] = []

    def fit(
        self,
        interactions: pd.DataFrame,
    ) -> "BPRRecommender":
        _validate_interactions(interactions)

        pairs = (
            interactions[
                ["user_idx", "artist_idx"]
            ]
            .drop_duplicates()
            .astype(
                {
                    "user_idx": "int64",
                    "artist_idx": "int64",
                }
            )
        )

        users = pairs["user_idx"].to_numpy(
            dtype=np.int64
        )

        items = pairs["artist_idx"].to_numpy(
            dtype=np.int64
        )

        if len(users) == 0:
            raise ValueError(
                "No positive interactions available for BPR."
            )

        self.n_users_ = int(users.max()) + 1
        self.n_items_ = int(items.max()) + 1

        self.known_user_mask_ = np.zeros(
            self.n_users_,
            dtype=bool,
        )

        self.known_item_mask_ = np.zeros(
            self.n_items_,
            dtype=bool,
        )

        self.known_user_mask_[np.unique(users)] = True
        self.known_item_mask_[np.unique(items)] = True

        self.known_item_ids_ = np.flatnonzero(
            self.known_item_mask_
        ).astype(np.int64)

        self.seen_matrix_ = np.zeros(
            (self.n_users_, self.n_items_),
            dtype=bool,
        )

        self.seen_matrix_[users, items] = True

        rng = np.random.default_rng(self.seed)

        self.user_factors_ = rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(
                self.n_users_,
                self.n_factors,
            ),
        ).astype(np.float32)

        self.item_factors_ = rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(
                self.n_items_,
                self.n_factors,
            ),
        ).astype(np.float32)

        self.item_bias_ = np.zeros(
            self.n_items_,
            dtype=np.float32,
        )

        n_positive = len(users)

        self.training_history_ = []

        for epoch in range(self.epochs):
            if self.max_samples_per_epoch is None:
                n_samples = n_positive
            else:
                n_samples = min(
                    n_positive,
                    self.max_samples_per_epoch,
                )

            if n_samples == n_positive:
                sampled_rows = rng.permutation(
                    n_positive
                )
            else:
                sampled_rows = rng.choice(
                    n_positive,
                    size=n_samples,
                    replace=False,
                )

            epoch_loss_sum = 0.0
            processed = 0

            for start in range(
                0,
                n_samples,
                self.batch_size,
            ):
                end = min(
                    start + self.batch_size,
                    n_samples,
                )

                batch_rows = sampled_rows[start:end]

                batch_users = users[batch_rows]
                positive_items = items[batch_rows]

                negative_items = rng.choice(
                    self.known_item_ids_,
                    size=len(batch_users),
                    replace=True,
                )

                invalid = self.seen_matrix_[
                    batch_users,
                    negative_items,
                ]

                # Rejection sampling until every negative item is
                # unobserved for its corresponding user.
                while np.any(invalid):
                    negative_items[invalid] = rng.choice(
                        self.known_item_ids_,
                        size=int(np.sum(invalid)),
                        replace=True,
                    )

                    invalid = self.seen_matrix_[
                        batch_users,
                        negative_items,
                    ]

                # Advanced indexing returns copies. This is useful here:
                # all gradients in the mini-batch are computed from the
                # same parameter state before accumulated updates.
                p_u = self.user_factors_[
                    batch_users
                ].copy()

                q_i = self.item_factors_[
                    positive_items
                ].copy()

                q_j = self.item_factors_[
                    negative_items
                ].copy()

                b_i = self.item_bias_[
                    positive_items
                ].copy()

                b_j = self.item_bias_[
                    negative_items
                ].copy()

                x_uij = (
                    b_i
                    - b_j
                    + np.einsum(
                        "ij,ij->i",
                        p_u,
                        q_i - q_j,
                    )
                )

                # derivative of -log(sigmoid(x)) with respect to x
                gradient_weight = expit(-x_uij).astype(
                    np.float32
                )

                reg = np.float32(
                    self.regularization
                )

                lr = np.float32(
                    self.learning_rate
                )

                user_delta = lr * (
                    gradient_weight[:, None]
                    * (q_i - q_j)
                    - reg * p_u
                )

                positive_delta = lr * (
                    gradient_weight[:, None]
                    * p_u
                    - reg * q_i
                )

                negative_delta = lr * (
                    -gradient_weight[:, None]
                    * p_u
                    - reg * q_j
                )

                positive_bias_delta = lr * (
                    gradient_weight
                    - reg * b_i
                )

                negative_bias_delta = lr * (
                    -gradient_weight
                    - reg * b_j
                )

                np.add.at(
                    self.user_factors_,
                    batch_users,
                    user_delta,
                )

                np.add.at(
                    self.item_factors_,
                    positive_items,
                    positive_delta,
                )

                np.add.at(
                    self.item_factors_,
                    negative_items,
                    negative_delta,
                )

                np.add.at(
                    self.item_bias_,
                    positive_items,
                    positive_bias_delta,
                )

                np.add.at(
                    self.item_bias_,
                    negative_items,
                    negative_bias_delta,
                )

                batch_loss = np.logaddexp(
                    0.0,
                    -x_uij,
                )

                epoch_loss_sum += float(
                    batch_loss.sum()
                )

                processed += len(batch_users)

            mean_loss = (
                epoch_loss_sum / processed
                if processed > 0
                else float("nan")
            )

            self.training_history_.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_pairwise_loss": mean_loss,
                    "samples": float(processed),
                }
            )

            if self.verbose:
                print(
                    f"BPR epoch "
                    f"{epoch + 1:02d}/{self.epochs:02d} "
                    f"- samples={processed:,} "
                    f"- pairwise_loss={mean_loss:.6f}"
                )

        return self

    def score_all_items(
        self,
        user_idx: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        if self.user_factors_ is None:
            raise RuntimeError(
                "BPR model has not been fitted."
            )

        if (
            user_idx < 0
            or user_idx >= self.n_users_
            or not self.known_user_mask_[user_idx]
        ):
            raise ValueError(
                f"Unknown user index: {user_idx}"
            )

        user_vector = self.user_factors_[
            user_idx
        ]

        scores = (
            self.item_bias_.astype(np.float64)
            + self.item_factors_.astype(
                np.float64
            )
            @ user_vector.astype(np.float64)
        )

        # Do not recommend items that were never observed during
        # training. Collaborative filtering cannot estimate these
        # future cold-start items meaningfully.
        scores[
            ~self.known_item_mask_
        ] = -np.inf

        if exclude_seen:
            scores[
                self.seen_matrix_[user_idx]
            ] = -np.inf

        return scores

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        scores = self.score_all_items(
            user_idx=user_idx,
            exclude_seen=exclude_seen,
        )

        return _top_k_from_scores(
            scores,
            k,
        )
