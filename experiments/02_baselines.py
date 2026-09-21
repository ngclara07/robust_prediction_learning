from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from robust_prediction_learning.metrics import (
    aggregate_ranking_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from robust_prediction_learning.recommenders import (
    BPRRecommender,
    PopularityRecommender,
)
from robust_prediction_learning.utils import (
    ensure_dir,
    save_json,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
RESULT_TABLE_DIR = ROOT / "results" / "tables"
RESULT_FIGURE_DIR = ROOT / "results" / "figures"

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
VALIDATION_FILE = (
    PROCESSED_DIR / "validation.parquet"
)
TEST_FILE = PROCESSED_DIR / "test.parquet"

DEFAULT_SEED = 42
DEFAULT_K = 20


def load_unique_pairs(
    path: Path,
) -> pd.DataFrame:
    """
    Load only the columns needed for recommendation and collapse
    repeated listening events into unique implicit user-item positives.
    """
    print(f"Loading {path.name}...")

    df = pd.read_parquet(
        path,
        columns=[
            "user_idx",
            "artist_idx",
        ],
    )

    before = len(df)

    pairs = (
        df.drop_duplicates(
            subset=[
                "user_idx",
                "artist_idx",
            ]
        )
        .astype(
            {
                "user_idx": "int32",
                "artist_idx": "int32",
            }
        )
        .reset_index(drop=True)
    )

    print(
        f"  events={before:,} "
        f"-> unique user-artist pairs="
        f"{len(pairs):,}"
    )

    return pairs


def evaluate_model(
    model: Any,
    future_pairs: pd.DataFrame,
    k: int = 20,
    max_users: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """
    Evaluate a fitted recommender using full-catalog top-K ranking.

    Evaluation protocol
    -------------------
    1. Future interactions are treated as implicit relevant items.
    2. Items already observed during model fitting are excluded.
    3. Future items never observed anywhere during model fitting are
       excluded because collaborative filtering cannot estimate them.
    4. Users without at least one evaluable future item are omitted.
    5. Metrics are macro-averaged over evaluable users.
    """
    future_by_user = {
        int(user_idx): np.unique(
            group["artist_idx"].to_numpy(
                dtype=np.int64
            )
        )
        for user_idx, group
        in future_pairs.groupby(
            "user_idx",
            sort=False,
        )
    }

    candidate_users: list[int] = []

    for user_idx in future_by_user:
        if (
            user_idx < 0
            or user_idx >= model.n_users_
        ):
            continue

        if not model.known_user_mask_[
            user_idx
        ]:
            continue

        candidate_users.append(
            user_idx
        )

    candidate_users = sorted(
        candidate_users
    )

    if (
        max_users is not None
        and len(candidate_users)
        > max_users
    ):
        rng = np.random.default_rng(seed)

        candidate_users = sorted(
            rng.choice(
                candidate_users,
                size=max_users,
                replace=False,
            ).tolist()
        )

    per_user_metrics: list[
        dict[str, float]
    ] = []

    skipped_no_relevant = 0
    evaluated_users = 0

    for position, user_idx in enumerate(
        candidate_users,
        start=1,
    ):
        future_items = future_by_user[
            user_idx
        ]

        # Keep only item IDs representable by the trained model.
        in_range = (
            (future_items >= 0)
            & (
                future_items
                < model.n_items_
            )
        )

        relevant = future_items[in_range]

        if len(relevant) == 0:
            skipped_no_relevant += 1
            continue

        relevant = relevant[
            model.known_item_mask_[
                relevant
            ]
        ]

        if len(relevant) == 0:
            skipped_no_relevant += 1
            continue

        # Evaluate discovery of future items not already observed
        # in the historical training window.
        relevant = relevant[
            ~model.seen_matrix_[
                user_idx,
                relevant,
            ]
        ]

        if len(relevant) == 0:
            skipped_no_relevant += 1
            continue

        recommendations = (
            model.recommend(
                user_idx=user_idx,
                k=k,
                exclude_seen=True,
            )
        )

        metrics = {
            f"precision_at_{k}":
                precision_at_k(
                    recommendations,
                    relevant,
                    k,
                ),
            f"recall_at_{k}":
                recall_at_k(
                    recommendations,
                    relevant,
                    k,
                ),
            f"ndcg_at_{k}":
                ndcg_at_k(
                    recommendations,
                    relevant,
                    k,
                ),
            f"mrr_at_{k}":
                mrr_at_k(
                    recommendations,
                    relevant,
                    k,
                ),
        }

        per_user_metrics.append(
            metrics
        )

        evaluated_users += 1

        if (
            evaluated_users % 100 == 0
        ):
            print(
                f"  evaluated "
                f"{evaluated_users:,} users..."
            )

    aggregated = (
        aggregate_ranking_metrics(
            per_user_metrics
        )
    )

    aggregated[
        "evaluated_users"
    ] = float(evaluated_users)

    aggregated[
        "skipped_users_no_evaluable_relevant_items"
    ] = float(
        skipped_no_relevant
    )

    return aggregated


def get_search_space(
    quick: bool,
) -> list[dict[str, Any]]:
    """
    Small deterministic BPR hyperparameter search.

    This deliberately avoids a large optimization sweep because the
    project studies robust prediction augmentation, not recommender
    architecture search.
    """
    if quick:
        return [
            {
                "n_factors": 32,
                "learning_rate": 0.05,
                "regularization": 0.005,
                "epochs": 2,
                "batch_size": 8192,
                "max_samples_per_epoch":
                    100_000,
            },
            {
                "n_factors": 64,
                "learning_rate": 0.05,
                "regularization": 0.005,
                "epochs": 2,
                "batch_size": 8192,
                "max_samples_per_epoch":
                    100_000,
            },
        ]

    return [
        {
            "n_factors": 32,
            "learning_rate": 0.05,
            "regularization": 0.005,
            "epochs": 5,
            "batch_size": 8192,
            "max_samples_per_epoch":
                500_000,
        },
        {
            "n_factors": 64,
            "learning_rate": 0.05,
            "regularization": 0.005,
            "epochs": 5,
            "batch_size": 8192,
            "max_samples_per_epoch":
                500_000,
        },
        {
            "n_factors": 64,
            "learning_rate": 0.025,
            "regularization": 0.01,
            "epochs": 6,
            "batch_size": 8192,
            "max_samples_per_epoch":
                750_000,
        },
    ]


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    print()
    print(model_name)
    print("-" * len(model_name))

    for key, value in metrics.items():
        if "users" in key:
            print(
                f"{key:45s}: "
                f"{int(value):,}"
            )
        else:
            print(
                f"{key:45s}: "
                f"{value:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Popularity and BPR "
            "recommendation baselines."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a reduced sanity experiment "
            "before the full research run."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError(
            "--k must be positive."
        )

    set_seed(args.seed)

    ensure_dir(RESULT_TABLE_DIR)
    ensure_dir(RESULT_FIGURE_DIR)

    print("=" * 70)
    print("MILESTONE 2: BASELINE RECOMMENDATION")
    print("=" * 70)

    print(
        f"Mode: {'QUICK' if args.quick else 'FULL'}"
    )

    print(
        f"Ranking cutoff K: {args.k}"
    )

    # ========================================================
    # LOAD PROCESSED DATA
    # ========================================================

    train = load_unique_pairs(
        TRAIN_FILE
    )

    validation = load_unique_pairs(
        VALIDATION_FILE
    )

    test = load_unique_pairs(
        TEST_FILE
    )

    print()
    print("Unique interaction summary")
    print("-" * 40)
    print(
        f"Train      : {len(train):,}"
    )
    print(
        f"Validation : "
        f"{len(validation):,}"
    )
    print(
        f"Test       : {len(test):,}"
    )

    # ========================================================
    # POPULARITY VALIDATION BASELINE
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION: POPULARITY BASELINE")
    print("=" * 70)

    popularity_train = (
        PopularityRecommender()
    )

    popularity_train.fit(train)

    validation_user_limit = (
        100
        if args.quick
        else None
    )

    popularity_validation_metrics = (
        evaluate_model(
            model=popularity_train,
            future_pairs=validation,
            k=args.k,
            max_users=validation_user_limit,
            seed=args.seed,
        )
    )

    print_metrics(
        "Popularity validation",
        popularity_validation_metrics,
    )

    # ========================================================
    # BPR VALIDATION SEARCH
    # ========================================================

    search_space = get_search_space(
        args.quick
    )

    validation_rows: list[
        dict[str, Any]
    ] = []

    best_model: BPRRecommender | None = None
    best_config: dict[str, Any] | None = None
    best_ndcg = -np.inf

    print()
    print("=" * 70)
    print("VALIDATION: BPR SEARCH")
    print("=" * 70)

    for config_index, config in enumerate(
        search_space,
        start=1,
    ):
        print()
        print(
            f"BPR configuration "
            f"{config_index}/"
            f"{len(search_space)}"
        )

        print(
            json.dumps(
                config,
                indent=2,
            )
        )

        start_time = time.perf_counter()

        model = BPRRecommender(
            **config,
            seed=args.seed,
            verbose=True,
        )

        model.fit(train)

        metrics = evaluate_model(
            model=model,
            future_pairs=validation,
            k=args.k,
            max_users=validation_user_limit,
            seed=args.seed,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print_metrics(
            f"BPR config {config_index}",
            metrics,
        )

        print(
            f"elapsed_seconds"
            f"{'':30s}: "
            f"{elapsed:.2f}"
        )

        row = {
            "configuration":
                config_index,
            **config,
            **metrics,
            "elapsed_seconds":
                elapsed,
        }

        validation_rows.append(row)

        current_ndcg = metrics[
            f"ndcg_at_{args.k}"
        ]

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_model = model
            best_config = config.copy()

    validation_results = pd.DataFrame(
        validation_rows
    )

    validation_output = (
        RESULT_TABLE_DIR
        / "bpr_validation_search.csv"
    )

    validation_results.to_csv(
        validation_output,
        index=False,
    )

    assert best_config is not None

    print()
    print("=" * 70)
    print("SELECTED BPR CONFIGURATION")
    print("=" * 70)

    print(
        json.dumps(
            best_config,
            indent=2,
        )
    )

    save_json(
        {
            "selection_metric":
                f"ndcg_at_{args.k}",
            "selected_parameters":
                best_config,
            "validation_ndcg":
                best_ndcg,
            "quick_mode":
                args.quick,
        },
        RESULT_TABLE_DIR
        / "selected_bpr_config.json",
    )

    # ========================================================
    # FINAL FIT: TRAIN + VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL FIT: TRAIN + VALIDATION")
    print("=" * 70)

    train_validation = (
        pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "user_idx",
                "artist_idx",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Combined unique "
        f"user-artist pairs: "
        f"{len(train_validation):,}"
    )

    # --------------------------------------------------------
    # Final popularity model
    # --------------------------------------------------------

    final_popularity = (
        PopularityRecommender()
    )

    final_popularity.fit(
        train_validation
    )

    # --------------------------------------------------------
    # Final BPR model
    # --------------------------------------------------------

    final_bpr = BPRRecommender(
        **best_config,
        seed=args.seed,
        verbose=True,
    )

    final_bpr.fit(
        train_validation
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    # The final test is evaluated over every evaluable user,
    # even when --quick was used. Quick mode only reduces model
    # training/search cost.
    popularity_test_metrics = (
        evaluate_model(
            model=final_popularity,
            future_pairs=test,
            k=args.k,
            max_users=None,
            seed=args.seed,
        )
    )

    bpr_test_metrics = (
        evaluate_model(
            model=final_bpr,
            future_pairs=test,
            k=args.k,
            max_users=None,
            seed=args.seed,
        )
    )

    print_metrics(
        "Popularity test",
        popularity_test_metrics,
    )

    print_metrics(
        "BPR test",
        bpr_test_metrics,
    )

    # ========================================================
    # SAVE TEST RESULTS
    # ========================================================

    test_results = pd.DataFrame(
        [
            {
                "model":
                    "Popularity",
                **popularity_test_metrics,
            },
            {
                "model":
                    "BPR",
                **bpr_test_metrics,
            },
        ]
    )

    test_results_path = (
        RESULT_TABLE_DIR
        / "baseline_results.csv"
    )

    test_results.to_csv(
        test_results_path,
        index=False,
    )

    # ========================================================
    # FIGURE
    # ========================================================

    metric_columns = [
        f"recall_at_{args.k}",
        f"ndcg_at_{args.k}",
        f"mrr_at_{args.k}",
    ]

    plot_data = (
        test_results[
            [
                "model",
                *metric_columns,
            ]
        ]
        .melt(
            id_vars="model",
            var_name="metric",
            value_name="score",
        )
    )

    plot_data["metric"] = (
        plot_data["metric"]
        .str.replace(
            "_",
            " ",
            regex=False,
        )
        .str.upper()
    )

    sns.set_theme(
        style="whitegrid"
    )

    plt.figure(
        figsize=(9, 5)
    )

    ax = sns.barplot(
        data=plot_data,
        x="metric",
        y="score",
        hue="model",
    )

    ax.set_title(
        "Chronological Recommendation "
        "Baseline Performance"
    )

    ax.set_xlabel("")
    ax.set_ylabel("Score")

    plt.tight_layout()

    figure_path = (
        RESULT_FIGURE_DIR
        / "baseline_comparison.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 70)
    print("GENERATED OUTPUTS")
    print("=" * 70)

    print(validation_output)
    print(
        RESULT_TABLE_DIR
        / "selected_bpr_config.json"
    )
    print(test_results_path)
    print(figure_path)


if __name__ == "__main__":
    main()
