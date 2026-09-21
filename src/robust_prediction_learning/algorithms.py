from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def expected_mixture_loss(
    historical_weight: float,
    historical_loss: float,
    online_loss: float,
) -> float:
    """
    Expected loss of a probabilistic mixture of two experts.

    With probability alpha, follow the historical predictor.
    With probability 1-alpha, follow the online predictor.

    Parameters
    ----------
    historical_weight:
        alpha in [0, 1].

    historical_loss:
        Loss of the historical predictor.

    online_loss:
        Loss of the online predictor.
    """
    alpha = float(historical_weight)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "historical_weight must lie in [0, 1]."
        )

    if not 0.0 <= historical_loss <= 1.0:
        raise ValueError(
            "historical_loss must lie in [0, 1]."
        )

    if not 0.0 <= online_loss <= 1.0:
        raise ValueError(
            "online_loss must lie in [0, 1]."
        )

    return float(
        alpha * historical_loss
        + (1.0 - alpha) * online_loss
    )


def _top_k_indices(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return indices of the K highest finite scores."""
    if k <= 0:
        raise ValueError("k must be positive.")

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    finite = np.flatnonzero(
        np.isfinite(scores)
    )

    if len(finite) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    k = min(
        k,
        len(finite),
    )

    finite_scores = scores[finite]

    if len(finite) <= k:
        order = np.argsort(
            finite_scores
        )[::-1]

        return finite[
            order
        ].astype(np.int64)

    candidate_positions = np.argpartition(
        finite_scores,
        -k,
    )[-k:]

    candidates = finite[
        candidate_positions
    ]

    order = np.argsort(
        scores[candidates]
    )[::-1]

    return candidates[
        order
    ].astype(np.int64)


class ExponentialDecayOnlineExpert:
    """
    Lightweight adaptive recommender based on exponentially decayed
    recent interaction counts.

    The expert deliberately remains simple: the research problem is
    how to combine historical ML predictions with adaptive evidence,
    rather than how to build another complex recommender.

    Before any online observations are available, the expert falls
    back to global item popularity.

    Update rule
    -----------
        c_t = decay * c_{t-1} + x_t

    where x_t contains the item counts observed in the current
    interaction window.
    """

    def __init__(
        self,
        known_item_mask: np.ndarray,
        popularity_scores: np.ndarray,
        decay: float = 0.80,
        prior_strength: float = 250.0,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                "decay must lie in [0, 1)."
            )

        if prior_strength < 0.0:
            raise ValueError(
                "prior_strength cannot be negative."
            )

        known_item_mask = np.asarray(
            known_item_mask,
            dtype=bool,
        )

        popularity_scores = np.asarray(
            popularity_scores,
            dtype=np.float64,
        )

        if known_item_mask.ndim != 1:
            raise ValueError(
                "known_item_mask must be one-dimensional."
            )

        if popularity_scores.shape != known_item_mask.shape:
            raise ValueError(
                "popularity_scores and known_item_mask "
                "must have equal shape."
            )

        if np.any(
            popularity_scores < 0
        ):
            raise ValueError(
                "popularity_scores cannot be negative."
            )

        self.known_item_mask = (
            known_item_mask.copy()
        )

        self.decay = float(decay)

        self.prior_strength = float(
            prior_strength
        )

        prior = np.zeros(
            len(known_item_mask),
            dtype=np.float64,
        )

        prior[
            self.known_item_mask
        ] = popularity_scores[
            self.known_item_mask
        ]

        prior_sum = float(
            prior.sum()
        )

        if prior_sum <= 0.0:
            n_known = int(
                self.known_item_mask.sum()
            )

            if n_known == 0:
                raise ValueError(
                    "At least one known item is required."
                )

            prior[
                self.known_item_mask
            ] = 1.0 / n_known

        else:
            prior /= prior_sum

        self.prior_probabilities = prior

        self.counts = np.zeros(
            len(known_item_mask),
            dtype=np.float64,
        )

    def reset(self) -> None:
        """Reset all online observations."""
        self.counts.fill(0.0)

    def recommend(
        self,
        k: int = 20,
    ) -> np.ndarray:
        """
        Return the current top-K adaptive recommendation.

        The popularity prior behaves like a virtual historical window.
        """
        scores = (
            self.counts
            + self.prior_strength
            * self.prior_probabilities
        )

        scores = scores.copy()

        scores[
            ~self.known_item_mask
        ] = -np.inf

        return _top_k_indices(
            scores,
            k,
        )

    def update(
        self,
        item_counts: Mapping[
            int,
            int | float,
        ],
    ) -> None:
        """Observe one new temporal interaction window."""
        self.counts *= self.decay

        for item, count in item_counts.items():
            item = int(item)
            value = float(count)

            if value < 0:
                raise ValueError(
                    "Interaction counts cannot be negative."
                )

            if (
                0 <= item < len(self.counts)
                and self.known_item_mask[item]
            ):
                self.counts[item] += value

    def observed_profile(
        self,
        minimum_mass: float = 1e-12,
    ) -> dict[int, float]:
        """
        Return the exponentially decayed behavioral profile.

        The popularity prior is intentionally excluded because the
        shift signal should reflect only actual observed behavior.
        """
        positive = np.flatnonzero(
            self.counts
            > minimum_mass
        )

        return {
            int(item): float(
                self.counts[item]
            )
            for item in positive
        }

    @property
    def observed_mass(self) -> float:
        return float(
            self.counts.sum()
        )


class PredictionOnly:
    """Always trust the historical predictor."""

    name = "PredictionOnly"

    def reset(self) -> None:
        pass

    def historical_weight(
        self,
        shift_signal: float = 0.0,
    ) -> float:
        return 1.0

    def observe(
        self,
        historical_loss: float,
        online_loss: float,
    ) -> None:
        pass


class OnlineOnly:
    """Ignore the historical predictor completely."""

    name = "OnlineOnly"

    def reset(self) -> None:
        pass

    def historical_weight(
        self,
        shift_signal: float = 0.0,
    ) -> float:
        return 0.0

    def observe(
        self,
        historical_loss: float,
        online_loss: float,
    ) -> None:
        pass


class StaticMixture:
    """
    Fixed convex combination of historical and online experts.
    """

    name = "StaticMixture"

    def __init__(
        self,
        historical_weight: float,
    ) -> None:
        if not 0.0 <= historical_weight <= 1.0:
            raise ValueError(
                "historical_weight must lie in [0, 1]."
            )

        self.weight = float(
            historical_weight
        )

    def reset(self) -> None:
        pass

    def historical_weight(
        self,
        shift_signal: float = 0.0,
    ) -> float:
        return self.weight

    def observe(
        self,
        historical_loss: float,
        online_loss: float,
    ) -> None:
        pass


class Hedge:
    """
    Exponentially weighted expert aggregation.

    The two experts are:
        0. historical predictor
        1. adaptive online predictor

    Update rule
    -----------
        w_{i,t+1}
            proportional to
        w_{i,t} exp(-eta * loss_{i,t})

    Losses are assumed to lie in [0, 1].
    """

    name = "Hedge"

    def __init__(
        self,
        learning_rate: float = 1.0,
        initial_historical_weight: float = 0.5,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if not (
            0.0
            < initial_historical_weight
            < 1.0
        ):
            raise ValueError(
                "initial_historical_weight "
                "must lie strictly between 0 and 1."
            )

        self.learning_rate = float(
            learning_rate
        )

        self.initial_historical_weight = float(
            initial_historical_weight
        )

        self.log_weights = np.empty(
            2,
            dtype=np.float64,
        )

        self.reset()

    def reset(self) -> None:
        self.log_weights[0] = np.log(
            self.initial_historical_weight
        )

        self.log_weights[1] = np.log(
            1.0
            - self.initial_historical_weight
        )

    def _weights(self) -> np.ndarray:
        shifted = (
            self.log_weights
            - np.max(
                self.log_weights
            )
        )

        weights = np.exp(
            shifted
        )

        return weights / weights.sum()

    def historical_weight(
        self,
        shift_signal: float = 0.0,
    ) -> float:
        return float(
            self._weights()[0]
        )

    def observe(
        self,
        historical_loss: float,
        online_loss: float,
    ) -> None:
        losses = np.array(
            [
                historical_loss,
                online_loss,
            ],
            dtype=np.float64,
        )

        if np.any(
            (losses < 0.0)
            | (losses > 1.0)
        ):
            raise ValueError(
                "Hedge losses must lie in [0, 1]."
            )

        self.log_weights -= (
            self.learning_rate
            * losses
        )


class ShiftAwareAdaptiveTrust:
    """
    Adaptive trust algorithm using both:

    1. behavioral distribution shift, and
    2. observed relative expert performance.

    Historical trust before round t is

        alpha_t =
            sigmoid(
                logit(alpha_0)
                - gamma_shift * s_t
                - gamma_disadvantage * d_t
            )

    where:

        alpha_0
            = baseline historical trust,

        s_t
            = behavioral shift signal in [0, 1],

        d_t
            = exponentially smoothed historical disadvantage:

              loss_historical - loss_online.

    Positive d_t means the historical model has recently performed
    worse than the adaptive expert and therefore decreases trust.

    Negative d_t increases historical trust.

    Crucially, the shift signal used for round t must be computed only
    from observations available before round t.
    """

    name = "ShiftAwareAdaptiveTrust"

    def __init__(
        self,
        base_trust: float = 0.80,
        gamma_shift: float = 2.0,
        gamma_disadvantage: float = 2.0,
        beta: float = 0.25,
        min_trust: float = 0.02,
        max_trust: float = 0.98,
    ) -> None:
        if not 0.0 < base_trust < 1.0:
            raise ValueError(
                "base_trust must lie strictly between 0 and 1."
            )

        if gamma_shift < 0.0:
            raise ValueError(
                "gamma_shift cannot be negative."
            )

        if gamma_disadvantage < 0.0:
            raise ValueError(
                "gamma_disadvantage cannot be negative."
            )

        if not 0.0 < beta <= 1.0:
            raise ValueError(
                "beta must lie in (0, 1]."
            )

        if not (
            0.0
            <= min_trust
            < max_trust
            <= 1.0
        ):
            raise ValueError(
                "Require 0 <= min_trust "
                "< max_trust <= 1."
            )

        self.base_trust = float(
            base_trust
        )

        self.gamma_shift = float(
            gamma_shift
        )

        self.gamma_disadvantage = float(
            gamma_disadvantage
        )

        self.beta = float(beta)

        self.min_trust = float(
            min_trust
        )

        self.max_trust = float(
            max_trust
        )

        self.base_logit = float(
            np.log(
                self.base_trust
                / (
                    1.0
                    - self.base_trust
                )
            )
        )

        self.disadvantage_ema = 0.0

    def reset(self) -> None:
        self.disadvantage_ema = 0.0

    def historical_weight(
        self,
        shift_signal: float = 0.0,
    ) -> float:
        shift = float(
            np.clip(
                shift_signal,
                0.0,
                1.0,
            )
        )

        logit = (
            self.base_logit
            - self.gamma_shift
            * shift
            - self.gamma_disadvantage
            * self.disadvantage_ema
        )

        logit = float(
            np.clip(
                logit,
                -40.0,
                40.0,
            )
        )

        alpha = (
            1.0
            / (
                1.0
                + np.exp(-logit)
            )
        )

        return float(
            np.clip(
                alpha,
                self.min_trust,
                self.max_trust,
            )
        )

    def observe(
        self,
        historical_loss: float,
        online_loss: float,
    ) -> None:
        if not (
            0.0
            <= historical_loss
            <= 1.0
        ):
            raise ValueError(
                "historical_loss must lie in [0, 1]."
            )

        if not (
            0.0
            <= online_loss
            <= 1.0
        ):
            raise ValueError(
                "online_loss must lie in [0, 1]."
            )

        disadvantage = (
            historical_loss
            - online_loss
        )

        self.disadvantage_ema = (
            (
                1.0
                - self.beta
            )
            * self.disadvantage_ema
            + self.beta
            * disadvantage
        )
