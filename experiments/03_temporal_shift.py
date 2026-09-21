from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    pearsonr,
    spearmanr,
)

from robust_prediction_learning.metrics import (
    ndcg_at_k,
)
from robust_prediction_learning.recommenders import (
    BPRRecommender,
    PopularityRecommender,
)
from robust_prediction_learning.shift import (
    build_user_item_counts,
    event_mass_at_k,
    filter_counts_by_known_mask,
    fraction_events_not_in_set,
    fraction_events_unknown_by_mask,
    jensen_shannon_from_counts,
    make_equal_interaction_windows,
)
from robust_prediction_learning.utils import (
    ensure_dir,
    save_json,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

RESULT_TABLE_DIR = (
    ROOT
    / "results"
    / "tables"
)

RESULT_FIGURE_DIR = (
    ROOT
    / "results"
    / "figures"
)

TRAIN_FILE = (
    PROCESSED_DIR
    / "train.parquet"
)

VALIDATION_FILE = (
    PROCESSED_DIR
    / "validation.parquet"
)

BPR_CONFIG_FILE = (
    RESULT_TABLE_DIR
    / "selected_bpr_config.json"
)

DEFAULT_WINDOW_SIZE = 250
DEFAULT_MIN_WINDOWS = 3
DEFAULT_K = 20
DEFAULT_SEED = 42


def load_interactions(
    path: Path,
) -> pd.DataFrame:
    print(
        f"Loading {path.name}..."
    )

    data = pd.read_parquet(
        path,
        columns=[
            "user_idx",
            "artist_idx",
            "timestamp",
        ],
    )

    data["user_idx"] = (
        data["user_idx"]
        .astype("int32")
    )

    data["artist_idx"] = (
        data["artist_idx"]
        .astype("int32")
    )

    data = data.sort_values(
        [
            "user_idx",
            "timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)

    print(
        f"  interactions: "
        f"{len(data):,}"
    )

    print(
        f"  users: "
        f"{data['user_idx'].nunique():,}"
    )

    print(
        f"  artists: "
        f"{data['artist_idx'].nunique():,}"
    )

    return data


def load_bpr_parameters(
    quick: bool,
) -> dict[str, Any]:
    if not BPR_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "Missing selected BPR configuration: "
            f"{BPR_CONFIG_FILE}"
        )

    with BPR_CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    parameters = dict(
        config[
            "selected_parameters"
        ]
    )

    if quick:
        parameters[
            "epochs"
        ] = min(
            int(
                parameters.get(
                    "epochs",
                    2,
                )
            ),
            2,
        )

        parameters[
            "max_samples_per_epoch"
        ] = 100_000

    return parameters


def safe_spearman(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[float, float]:
    x_array = np.asarray(
        x,
        dtype=np.float64,
    )

    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[mask]
    y_array = y_array[mask]

    if len(x_array) < 3:
        return (
            float("nan"),
            float("nan"),
        )

    if (
        np.all(
            x_array
            == x_array[0]
        )
        or np.all(
            y_array
            == y_array[0]
        )
    ):
        return (
            float("nan"),
            float("nan"),
        )

    result = spearmanr(
        x_array,
        y_array,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def safe_pearson(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[float, float]:
    x_array = np.asarray(
        x,
        dtype=np.float64,
    )

    y_array = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
    )

    x_array = x_array[mask]
    y_array = y_array[mask]

    if len(x_array) < 3:
        return (
            float("nan"),
            float("nan"),
        )

    if (
        np.all(
            x_array
            == x_array[0]
        )
        or np.all(
            y_array
            == y_array[0]
        )
    ):
        return (
            float("nan"),
            float("nan"),
        )

    result = pearsonr(
        x_array,
        y_array,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def create_window_metrics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    bpr: BPRRecommender,
    popularity: PopularityRecommender,
    window_size: int,
    min_windows: int,
    k: int,
    max_users: int | None,
    seed: int,
) -> pd.DataFrame:
    """
    Measure behavioral distribution shift and frozen-predictor quality.

    The BPR and popularity models remain fixed throughout the validation
    period. Therefore changes in prediction quality reflect changes in
    future user behavior rather than online model updates.
    """
    print()
    print(
        "Constructing historical "
        "user distributions..."
    )

    historical_profiles = (
        build_user_item_counts(
            train
        )
    )

    history_end = (
        train
        .groupby(
            "user_idx"
        )["timestamp"]
        .max()
        .to_dict()
    )

    validation_users = np.array(
        sorted(
            set(
                validation[
                    "user_idx"
                ].unique()
            )
            & set(
                historical_profiles.keys()
            )
        ),
        dtype=np.int64,
    )

    if (
        max_users is not None
        and len(validation_users)
        > max_users
    ):
        rng = np.random.default_rng(
            seed
        )

        validation_users = np.sort(
            rng.choice(
                validation_users,
                size=max_users,
                replace=False,
            )
        )

    print(
        f"Candidate users: "
        f"{len(validation_users):,}"
    )

    validation_by_user = {
        int(user_idx): group
        for user_idx, group
        in validation[
            validation[
                "user_idx"
            ].isin(
                validation_users
            )
        ].groupby(
            "user_idx",
            sort=False,
        )
    }

    rows: list[
        dict[str, Any]
    ] = []

    users_analyzed = 0
    users_insufficient_windows = 0

    for user_position, user_idx in enumerate(
        validation_users,
        start=1,
    ):
        user_idx = int(
            user_idx
        )

        user_future = (
            validation_by_user.get(
                user_idx
            )
        )

        if user_future is None:
            continue

        windows = (
            make_equal_interaction_windows(
                user_future,
                window_size=window_size,
            )
        )

        if len(windows) < min_windows:
            users_insufficient_windows += 1
            continue

        historical_counts = (
            historical_profiles[
                user_idx
            ]
        )

        historical_items = set(
            historical_counts.keys()
        )

        bpr_recommendations = (
            bpr.recommend(
                user_idx=user_idx,
                k=k,
                exclude_seen=False,
            )
        )

        popularity_recommendations = (
            popularity.recommend(
                user_idx=user_idx,
                k=k,
                exclude_seen=False,
            )
        )

        previous_known_counts: (
            dict[int, int]
            | None
        ) = None

        user_history_end = (
            pd.Timestamp(
                history_end[
                    user_idx
                ]
            )
        )

        for window in windows:
            full_counts = (
                window.item_counts
            )

            known_counts = (
                filter_counts_by_known_mask(
                    full_counts,
                    bpr.known_item_mask_,
                )
            )

            if not known_counts:
                continue

            js_to_history_all = (
                jensen_shannon_from_counts(
                    historical_counts,
                    full_counts,
                )
            )

            js_to_history_known = (
                jensen_shannon_from_counts(
                    historical_counts,
                    known_counts,
                )
            )

            if (
                previous_known_counts
                is None
            ):
                js_to_previous = (
                    float("nan")
                )
            else:
                js_to_previous = (
                    jensen_shannon_from_counts(
                        previous_known_counts,
                        known_counts,
                    )
                )

            bpr_mass_all = (
                event_mass_at_k(
                    bpr_recommendations,
                    full_counts,
                    k=k,
                )
            )

            bpr_mass_known = (
                event_mass_at_k(
                    bpr_recommendations,
                    known_counts,
                    k=k,
                )
            )

            popularity_mass_all = (
                event_mass_at_k(
                    popularity_recommendations,
                    full_counts,
                    k=k,
                )
            )

            popularity_mass_known = (
                event_mass_at_k(
                    popularity_recommendations,
                    known_counts,
                    k=k,
                )
            )

            relevant_known = set(
                known_counts.keys()
            )

            bpr_ndcg = (
                ndcg_at_k(
                    bpr_recommendations,
                    relevant_known,
                    k=k,
                )
            )

            popularity_ndcg = (
                ndcg_at_k(
                    popularity_recommendations,
                    relevant_known,
                    k=k,
                )
            )

            user_novel_fraction = (
                fraction_events_not_in_set(
                    full_counts,
                    historical_items,
                )
            )

            global_cold_start_fraction = (
                fraction_events_unknown_by_mask(
                    full_counts,
                    bpr.known_item_mask_,
                )
            )

            elapsed_days = float(
                (
                    window.end_timestamp
                    - user_history_end
                ).total_seconds()
                / 86_400.0
            )

            rows.append(
                {
                    "user_idx":
                        user_idx,
                    "window_index":
                        window.window_index,
                    "window_number":
                        window.window_index
                        + 1,
                    "start_timestamp":
                        window.start_timestamp,
                    "end_timestamp":
                        window.end_timestamp,
                    "elapsed_days":
                        elapsed_days,
                    "n_interactions":
                        window.n_interactions,
                    "n_unique_artists":
                        len(full_counts),
                    "n_known_unique_artists":
                        len(known_counts),
                    "js_to_history_all":
                        js_to_history_all,
                    "js_to_history_known":
                        js_to_history_known,
                    "js_to_previous":
                        js_to_previous,
                    f"bpr_event_mass_at_{k}_all":
                        bpr_mass_all,
                    f"bpr_event_mass_at_{k}_known":
                        bpr_mass_known,
                    f"popularity_event_mass_at_{k}_all":
                        popularity_mass_all,
                    f"popularity_event_mass_at_{k}_known":
                        popularity_mass_known,
                    f"bpr_ndcg_at_{k}":
                        bpr_ndcg,
                    f"popularity_ndcg_at_{k}":
                        popularity_ndcg,
                    "user_novel_event_fraction":
                        user_novel_fraction,
                    "global_cold_start_fraction":
                        global_cold_start_fraction,
                }
            )

            previous_known_counts = (
                known_counts
            )

        users_analyzed += 1

        if (
            users_analyzed % 100
            == 0
        ):
            print(
                f"  analyzed "
                f"{users_analyzed:,} users..."
            )

    print()
    print(
        f"Users analyzed: "
        f"{users_analyzed:,}"
    )

    print(
        "Users skipped because fewer than "
        f"{min_windows} complete windows: "
        f"{users_insufficient_windows:,}"
    )

    if not rows:
        raise RuntimeError(
            "No valid temporal windows were generated. "
            "Try reducing --window-size or --min-windows."
        )

    result = pd.DataFrame(
        rows
    )

    primary_quality_col = (
        f"bpr_event_mass_at_{k}_known"
    )

    result[
        "bpr_quality_change_from_first"
    ] = (
        result
        .groupby(
            "user_idx"
        )[primary_quality_col]
        .transform(
            lambda values:
                values
                - values.iloc[0]
        )
    )

    result[
        "shift_change_from_first"
    ] = (
        result
        .groupby(
            "user_idx"
        )["js_to_history_known"]
        .transform(
            lambda values:
                values
                - values.iloc[0]
        )
    )

    return result


def compute_per_user_correlations(
    window_metrics: pd.DataFrame,
    quality_col: str,
) -> pd.DataFrame:
    rows: list[
        dict[str, float | int]
    ] = []

    for user_idx, group in (
        window_metrics
        .groupby(
            "user_idx"
        )
    ):
        rho, p_value = (
            safe_spearman(
                group[
                    "js_to_history_known"
                ],
                group[
                    quality_col
                ],
            )
        )

        rows.append(
            {
                "user_idx":
                    int(user_idx),
                "n_windows":
                    int(len(group)),
                "spearman_rho":
                    rho,
                "spearman_p_value":
                    p_value,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_shift_bins(
    window_metrics: pd.DataFrame,
    quality_col: str,
) -> pd.DataFrame:
    data = (
        window_metrics[
            [
                "js_to_history_known",
                quality_col,
            ]
        ]
        .dropna()
        .copy()
    )

    if len(data) < 5:
        return pd.DataFrame()

    data[
        "shift_bin"
    ] = pd.qcut(
        data[
            "js_to_history_known"
        ],
        q=5,
        duplicates="drop",
    )

    grouped = (
        data
        .groupby(
            "shift_bin",
            observed=True,
        )
        .agg(
            shift_mean=(
                "js_to_history_known",
                "mean",
            ),
            shift_min=(
                "js_to_history_known",
                "min",
            ),
            shift_max=(
                "js_to_history_known",
                "max",
            ),
            quality_mean=(
                quality_col,
                "mean",
            ),
            quality_std=(
                quality_col,
                "std",
            ),
            count=(
                quality_col,
                "size",
            ),
        )
        .reset_index()
    )

    grouped[
        "quality_sem"
    ] = (
        grouped[
            "quality_std"
        ]
        / np.sqrt(
            grouped[
                "count"
            ]
        )
    )

    grouped[
        "shift_bin"
    ] = (
        grouped[
            "shift_bin"
        ]
        .astype(str)
    )

    return grouped


def save_figures(
    window_metrics: pd.DataFrame,
    per_user_correlations: pd.DataFrame,
    shift_bins: pd.DataFrame,
    k: int,
) -> None:
    sns.set_theme(
        style="whitegrid"
    )

    quality_col = (
        f"bpr_event_mass_at_{k}_known"
    )

    popularity_col = (
        f"popularity_event_mass_at_{k}_known"
    )

    # ========================================================
    # FIGURE 1: DISTRIBUTION OF BEHAVIORAL SHIFT
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        window_metrics[
            "js_to_history_known"
        ],
        bins=40,
        kde=True,
    )

    plt.xlabel(
        "Jensen-Shannon divergence "
        "from historical listening distribution"
    )

    plt.ylabel(
        "Number of equal-interaction windows"
    )

    plt.title(
        "Distribution of Temporal "
        "Behavioral Shift"
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_FIGURE_DIR
        / "temporal_shift_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 2: SHIFT VS BPR PREDICTION QUALITY
    # ========================================================

    finite = (
        window_metrics[
            [
                "js_to_history_known",
                quality_col,
            ]
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    plt.figure(
        figsize=(8, 6)
    )

    hexbin = plt.hexbin(
        finite[
            "js_to_history_known"
        ],
        finite[
            quality_col
        ],
        gridsize=35,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )

    colorbar = plt.colorbar(
        hexbin
    )

    colorbar.set_label(
        "log10(window count)"
    )

    if not shift_bins.empty:
        plt.plot(
            shift_bins[
                "shift_mean"
            ],
            shift_bins[
                "quality_mean"
            ],
            color="red",
            marker="o",
            linewidth=2.0,
            label="Shift-quintile mean",
        )

        plt.legend()

    plt.xlabel(
        "Jensen-Shannon divergence "
        "from historical distribution"
    )

    plt.ylabel(
        f"BPR event mass@{k}"
    )

    plt.title(
        "Behavioral Shift vs. "
        "Historical Predictor Quality"
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_FIGURE_DIR
        / "shift_vs_prediction_quality.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 3: PREDICTOR QUALITY ACROSS WINDOWS
    # ========================================================

    window_summary = (
        window_metrics
        .groupby(
            "window_number"
        )
        .agg(
            bpr_mean=(
                quality_col,
                "mean",
            ),
            bpr_std=(
                quality_col,
                "std",
            ),
            popularity_mean=(
                popularity_col,
                "mean",
            ),
            popularity_std=(
                popularity_col,
                "std",
            ),
            n_users=(
                "user_idx",
                "nunique",
            ),
        )
        .reset_index()
    )

    window_summary = (
        window_summary[
            window_summary[
                "n_users"
            ]
            >= 20
        ]
        .copy()
    )

    if not window_summary.empty:
        window_summary[
            "bpr_sem"
        ] = (
            window_summary[
                "bpr_std"
            ]
            / np.sqrt(
                window_summary[
                    "n_users"
                ]
            )
        )

        window_summary[
            "popularity_sem"
        ] = (
            window_summary[
                "popularity_std"
            ]
            / np.sqrt(
                window_summary[
                    "n_users"
                ]
            )
        )

        x = (
            window_summary[
                "window_number"
            ].to_numpy()
        )

        bpr_mean = (
            window_summary[
                "bpr_mean"
            ].to_numpy()
        )

        bpr_ci = (
            1.96
            * window_summary[
                "bpr_sem"
            ].to_numpy()
        )

        popularity_mean = (
            window_summary[
                "popularity_mean"
            ].to_numpy()
        )

        popularity_ci = (
            1.96
            * window_summary[
                "popularity_sem"
            ].to_numpy()
        )

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            x,
            bpr_mean,
            marker="o",
            label="BPR",
        )

        plt.fill_between(
            x,
            bpr_mean - bpr_ci,
            bpr_mean + bpr_ci,
            alpha=0.2,
        )

        plt.plot(
            x,
            popularity_mean,
            marker="o",
            label="Popularity",
        )

        plt.fill_between(
            x,
            popularity_mean
            - popularity_ci,
            popularity_mean
            + popularity_ci,
            alpha=0.2,
        )

        plt.xlabel(
            "Equal-interaction "
            "validation window"
        )

        plt.ylabel(
            f"Event mass@{k}"
        )

        plt.title(
            "Frozen Historical Predictor "
            "Quality Over Time"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULT_FIGURE_DIR
            / "predictor_quality_over_windows.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    # ========================================================
    # FIGURE 4: USER-LEVEL CORRELATION DISTRIBUTION
    # ========================================================

    correlations = (
        per_user_correlations[
            "spearman_rho"
        ]
        .dropna()
    )

    if not correlations.empty:
        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            correlations,
            bins=30,
            kde=True,
        )

        plt.axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        median_rho = float(
            correlations.median()
        )

        plt.axvline(
            median_rho,
            color="red",
            linestyle="-",
            linewidth=2.0,
            label=(
                "Median "
                f"$\\rho$={median_rho:.3f}"
            ),
        )

        plt.xlabel(
            "Per-user Spearman correlation:\n"
            "shift vs. BPR prediction quality"
        )

        plt.ylabel(
            "Number of users"
        )

        plt.title(
            "User-Level Relationship Between "
            "Behavioral Shift and Predictor Quality"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULT_FIGURE_DIR
            / "shift_by_user.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze temporal behavioral shift "
            "and degradation of a frozen "
            "historical recommender."
        )
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "Number of interactions in each "
            "equal-interaction temporal window."
        ),
    )

    parser.add_argument(
        "--min-windows",
        type=int,
        default=DEFAULT_MIN_WINDOWS,
        help=(
            "Minimum number of complete validation "
            "windows required for a user."
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

    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a reduced experiment for "
            "implementation validation."
        ),
    )

    args = parser.parse_args()

    if args.window_size <= 0:
        raise ValueError(
            "--window-size must be positive."
        )

    if args.min_windows < 2:
        raise ValueError(
            "--min-windows must be at least 2."
        )

    if args.k <= 0:
        raise ValueError(
            "--k must be positive."
        )

    set_seed(
        args.seed
    )

    ensure_dir(
        RESULT_TABLE_DIR
    )

    ensure_dir(
        RESULT_FIGURE_DIR
    )

    print(
        "=" * 72
    )

    print(
        "MILESTONE 3: TEMPORAL DISTRIBUTION SHIFT"
    )

    print(
        "=" * 72
    )

    print(
        f"Mode: "
        f"{'QUICK' if args.quick else 'FULL'}"
    )

    print(
        f"Window size: "
        f"{args.window_size:,} interactions"
    )

    print(
        f"Minimum windows/user: "
        f"{args.min_windows}"
    )

    print(
        f"Ranking cutoff K: "
        f"{args.k}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train = load_interactions(
        TRAIN_FILE
    )

    validation = load_interactions(
        VALIDATION_FILE
    )

    # ========================================================
    # LOAD FIXED BPR CONFIGURATION
    # ========================================================

    bpr_parameters = (
        load_bpr_parameters(
            quick=args.quick
        )
    )

    print()
    print(
        "Historical BPR parameters"
    )

    print(
        "-" * 40
    )

    print(
        json.dumps(
            bpr_parameters,
            indent=2,
        )
    )

    # ========================================================
    # FIT FROZEN HISTORICAL PREDICTORS
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "FIT FROZEN HISTORICAL PREDICTORS"
    )

    print(
        "=" * 72
    )

    popularity = (
        PopularityRecommender()
    )

    popularity.fit(
        train
    )

    bpr = BPRRecommender(
        **bpr_parameters,
        seed=args.seed,
        verbose=True,
    )

    bpr.fit(
        train
    )

    # ========================================================
    # TEMPORAL WINDOW ANALYSIS
    # ========================================================

    max_users = (
        100
        if args.quick
        and args.max_users is None
        else args.max_users
    )

    window_metrics = (
        create_window_metrics(
            train=train,
            validation=validation,
            bpr=bpr,
            popularity=popularity,
            window_size=args.window_size,
            min_windows=args.min_windows,
            k=args.k,
            max_users=max_users,
            seed=args.seed,
        )
    )

    quality_col = (
        f"bpr_event_mass_at_{args.k}_known"
    )

    ndcg_col = (
        f"bpr_ndcg_at_{args.k}"
    )

    # ========================================================
    # GLOBAL CORRELATIONS
    # ========================================================

    overall_spearman_mass = (
        safe_spearman(
            window_metrics[
                "js_to_history_known"
            ],
            window_metrics[
                quality_col
            ],
        )
    )

    overall_pearson_mass = (
        safe_pearson(
            window_metrics[
                "js_to_history_known"
            ],
            window_metrics[
                quality_col
            ],
        )
    )

    overall_spearman_ndcg = (
        safe_spearman(
            window_metrics[
                "js_to_history_known"
            ],
            window_metrics[
                ndcg_col
            ],
        )
    )

    # ========================================================
    # USER-LEVEL CORRELATIONS
    # ========================================================

    per_user_correlations = (
        compute_per_user_correlations(
            window_metrics,
            quality_col=quality_col,
        )
    )

    valid_user_rhos = (
        per_user_correlations[
            "spearman_rho"
        ]
        .dropna()
    )

    # ========================================================
    # SHIFT QUANTILES
    # ========================================================

    shift_bins = (
        build_shift_bins(
            window_metrics,
            quality_col=quality_col,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "window_size":
            args.window_size,
        "minimum_windows_per_user":
            args.min_windows,
        "ranking_k":
            args.k,
        "analyzed_users":
            int(
                window_metrics[
                    "user_idx"
                ].nunique()
            ),
        "analyzed_windows":
            int(
                len(
                    window_metrics
                )
            ),
        "mean_js_to_history_known":
            float(
                window_metrics[
                    "js_to_history_known"
                ].mean()
            ),
        "median_js_to_history_known":
            float(
                window_metrics[
                    "js_to_history_known"
                ].median()
            ),
        "mean_bpr_event_mass_known":
            float(
                window_metrics[
                    quality_col
                ].mean()
            ),
        "mean_bpr_ndcg":
            float(
                window_metrics[
                    ndcg_col
                ].mean()
            ),
        "mean_user_novel_event_fraction":
            float(
                window_metrics[
                    "user_novel_event_fraction"
                ].mean()
            ),
        "mean_global_cold_start_fraction":
            float(
                window_metrics[
                    "global_cold_start_fraction"
                ].mean()
            ),
        "overall_spearman_shift_vs_bpr_mass":
            overall_spearman_mass[0],
        "overall_spearman_p_value":
            overall_spearman_mass[1],
        "overall_pearson_shift_vs_bpr_mass":
            overall_pearson_mass[0],
        "overall_pearson_p_value":
            overall_pearson_mass[1],
        "overall_spearman_shift_vs_bpr_ndcg":
            overall_spearman_ndcg[0],
        "overall_spearman_ndcg_p_value":
            overall_spearman_ndcg[1],
        "users_with_valid_within_user_correlation":
            int(
                len(
                    valid_user_rhos
                )
            ),
        "median_user_spearman_rho":
            (
                float(
                    valid_user_rhos.median()
                )
                if not valid_user_rhos.empty
                else None
            ),
        "mean_user_spearman_rho":
            (
                float(
                    valid_user_rhos.mean()
                )
                if not valid_user_rhos.empty
                else None
            ),
        "fraction_users_negative_correlation":
            (
                float(
                    (
                        valid_user_rhos
                        < 0
                    ).mean()
                )
                if not valid_user_rhos.empty
                else None
            ),
    }

    # ========================================================
    # SAVE TABLES
    # ========================================================

    window_metrics_path = (
        RESULT_TABLE_DIR
        / "temporal_window_metrics.csv"
    )

    window_metrics.to_csv(
        window_metrics_path,
        index=False,
    )

    correlations_path = (
        RESULT_TABLE_DIR
        / "per_user_shift_correlations.csv"
    )

    per_user_correlations.to_csv(
        correlations_path,
        index=False,
    )

    shift_bins_path = (
        RESULT_TABLE_DIR
        / "shift_quality_bins.csv"
    )

    shift_bins.to_csv(
        shift_bins_path,
        index=False,
    )

    summary_path = (
        RESULT_TABLE_DIR
        / "temporal_shift_summary.json"
    )

    save_json(
        summary,
        summary_path,
    )

    # ========================================================
    # FIGURES
    # ========================================================

    save_figures(
        window_metrics=window_metrics,
        per_user_correlations=(
            per_user_correlations
        ),
        shift_bins=shift_bins,
        k=args.k,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "TEMPORAL SHIFT SUMMARY"
    )

    print(
        "=" * 72
    )

    for key, value in (
        summary.items()
    ):
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:50s}: "
                f"{value:.6f}"
            )
        else:
            print(
                f"{key:50s}: "
                f"{value}"
            )

    print()
    print(
        "=" * 72
    )

    print(
        "GENERATED OUTPUTS"
    )

    print(
        "=" * 72
    )

    print(
        window_metrics_path
    )

    print(
        correlations_path
    )

    print(
        shift_bins_path
    )

    print(
        summary_path
    )

    print(
        RESULT_FIGURE_DIR
        / "temporal_shift_distribution.png"
    )

    print(
        RESULT_FIGURE_DIR
        / "shift_vs_prediction_quality.png"
    )

    print(
        RESULT_FIGURE_DIR
        / "predictor_quality_over_windows.png"
    )

    print(
        RESULT_FIGURE_DIR
        / "shift_by_user.png"
    )


if __name__ == "__main__":
    main()
