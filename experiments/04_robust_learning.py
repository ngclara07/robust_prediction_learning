from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from robust_prediction_learning.algorithms import (
    ExponentialDecayOnlineExpert,
    Hedge,
    OnlineOnly,
    PredictionOnly,
    ShiftAwareAdaptiveTrust,
    StaticMixture,
    expected_mixture_loss,
)
from robust_prediction_learning.recommenders import (
    BPRRecommender,
    PopularityRecommender,
)
from robust_prediction_learning.shift import (
    build_user_item_counts,
    event_mass_at_k,
    filter_counts_by_known_mask,
    jensen_shannon_from_counts,
    make_equal_interaction_windows,
)
from robust_prediction_learning.utils import (
    ensure_dir,
    save_json,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
VALIDATION_FILE = PROCESSED_DIR / "validation.parquet"
TEST_FILE = PROCESSED_DIR / "test.parquet"

BPR_CONFIG_FILE = (
    TABLE_DIR
    / "selected_bpr_config.json"
)

DEFAULT_WINDOW_SIZE = 250
DEFAULT_MIN_WINDOWS = 3
DEFAULT_K = 20
DEFAULT_SEED = 42

DEFAULT_ONLINE_DECAY = 0.80


def load_interactions(
    path: Path,
) -> pd.DataFrame:
    print(f"Loading {path.name}...")

    df = pd.read_parquet(
        path,
        columns=[
            "user_idx",
            "artist_idx",
            "timestamp",
        ],
    )

    df["user_idx"] = df[
        "user_idx"
    ].astype("int32")

    df["artist_idx"] = df[
        "artist_idx"
    ].astype("int32")

    df = df.sort_values(
        [
            "user_idx",
            "timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)

    print(
        f"  interactions: {len(df):,}"
    )

    print(
        f"  users: "
        f"{df['user_idx'].nunique():,}"
    )

    return df


def load_bpr_parameters(
    quick: bool,
) -> dict[str, Any]:
    if not BPR_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "Missing BPR configuration: "
            f"{BPR_CONFIG_FILE}"
        )

    with BPR_CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    parameters = dict(
        config["selected_parameters"]
    )

    if quick:
        parameters["epochs"] = min(
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


def build_method(
    specification: dict[str, Any],
):
    family = specification["family"]

    params = specification.get(
        "params",
        {},
    )

    if family == "prediction":
        return PredictionOnly()

    if family == "online":
        return OnlineOnly()

    if family == "static":
        return StaticMixture(
            **params
        )

    if family == "hedge":
        return Hedge(
            **params
        )

    if family == "shift":
        return ShiftAwareAdaptiveTrust(
            **params
        )

    raise ValueError(
        f"Unknown method family: {family}"
    )


def validation_method_specs() -> list[
    dict[str, Any]
]:
    """
    Small pre-specified validation search.

    Parameter search is intentionally limited because this study is
    about robust prediction combination rather than extensive
    hyperparameter optimization.
    """
    specs: list[
        dict[str, Any]
    ] = [
        {
            "name": "PredictionOnly",
            "family": "prediction",
            "params": {},
        },
        {
            "name": "OnlineOnly",
            "family": "online",
            "params": {},
        },
    ]

    for alpha in [
        0.25,
        0.50,
        0.75,
    ]:
        specs.append(
            {
                "name":
                    f"Static_alpha_{alpha:.2f}",
                "family":
                    "static",
                "params":
                    {
                        "historical_weight":
                            alpha,
                    },
            }
        )

    for eta in [
        0.5,
        1.0,
        2.0,
    ]:
        specs.append(
            {
                "name":
                    f"Hedge_eta_{eta:.2f}",
                "family":
                    "hedge",
                "params":
                    {
                        "learning_rate":
                            eta,
                        "initial_historical_weight":
                            0.5,
                    },
            }
        )

    shift_configs = [
        {
            "base_trust": 0.75,
            "gamma_shift": 1.0,
            "gamma_disadvantage": 2.0,
            "beta": 0.25,
        },
        {
            "base_trust": 0.75,
            "gamma_shift": 2.0,
            "gamma_disadvantage": 2.0,
            "beta": 0.25,
        },
        {
            "base_trust": 0.90,
            "gamma_shift": 2.0,
            "gamma_disadvantage": 2.0,
            "beta": 0.25,
        },
        {
            "base_trust": 0.90,
            "gamma_shift": 4.0,
            "gamma_disadvantage": 2.0,
            "beta": 0.25,
        },
    ]

    for index, params in enumerate(
        shift_configs,
        start=1,
    ):
        specs.append(
            {
                "name":
                    f"ShiftAware_{index}",
                "family":
                    "shift",
                "params":
                    params,
            }
        )

    return specs


def selected_test_specs(
    selected_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "name": "PredictionOnly",
            "display_name":
                "Prediction Only",
            "family": "prediction",
            "params": {},
        },
        {
            "name": "OnlineOnly",
            "display_name":
                "Online Only",
            "family": "online",
            "params": {},
        },
        {
            "name": "StaticMixture",
            "display_name":
                "Static Mixture",
            "family": "static",
            "params":
                selected_parameters[
                    "static"
                ],
        },
        {
            "name": "Hedge",
            "display_name":
                "Hedge",
            "family": "hedge",
            "params":
                selected_parameters[
                    "hedge"
                ],
        },
        {
            "name":
                "ShiftAwareAdaptiveTrust",
            "display_name":
                "Shift-Aware Trust",
            "family": "shift",
            "params":
                selected_parameters[
                    "shift"
                ],
        },
    ]


def fit_historical_models(
    history: pd.DataFrame,
    bpr_parameters: dict[str, Any],
    seed: int,
) -> tuple[
    BPRRecommender,
    PopularityRecommender,
]:
    print()
    print(
        "Fitting historical popularity model..."
    )

    popularity = (
        PopularityRecommender()
    )

    popularity.fit(
        history
    )

    print(
        "Fitting historical BPR model..."
    )

    bpr = BPRRecommender(
        **bpr_parameters,
        seed=seed,
        verbose=True,
    )

    bpr.fit(
        history
    )

    return (
        bpr,
        popularity,
    )


def evaluate_sequential_stream(
    history: pd.DataFrame,
    stream: pd.DataFrame,
    bpr: BPRRecommender,
    popularity: PopularityRecommender,
    method_specs: list[
        dict[str, Any]
    ],
    window_size: int,
    min_windows: int,
    k: int,
    online_decay: float,
    max_users: int | None,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Sequentially evaluate expert-combination algorithms.

    At each round:

    1. predictions are produced using only past observations;
    2. current-window losses are evaluated;
    3. combination algorithms observe expert losses;
    4. the online expert is updated using the current window.

    This ordering prevents look-ahead leakage.
    """
    historical_profiles = (
        build_user_item_counts(
            history
        )
    )

    candidate_users = np.array(
        sorted(
            set(
                stream[
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
        and len(candidate_users)
        > max_users
    ):
        rng = np.random.default_rng(
            seed
        )

        candidate_users = np.sort(
            rng.choice(
                candidate_users,
                size=max_users,
                replace=False,
            )
        )

    stream_by_user = {
        int(user_idx): group
        for user_idx, group
        in stream[
            stream[
                "user_idx"
            ].isin(
                candidate_users
            )
        ].groupby(
            "user_idx",
            sort=False,
        )
    }

    window_rows: list[
        dict[str, Any]
    ] = []

    user_rows: list[
        dict[str, Any]
    ] = []

    analyzed_users = 0
    skipped_users = 0

    for user_idx in candidate_users:
        user_idx = int(
            user_idx
        )

        user_stream = (
            stream_by_user.get(
                user_idx
            )
        )

        if user_stream is None:
            continue

        windows = (
            make_equal_interaction_windows(
                user_stream,
                window_size=window_size,
            )
        )

        if len(windows) < min_windows:
            skipped_users += 1
            continue

        historical_counts = (
            historical_profiles[
                user_idx
            ]
        )

        historical_recommendations = (
            bpr.recommend(
                user_idx=user_idx,
                k=k,
                exclude_seen=False,
            )
        )

        online_expert = (
            ExponentialDecayOnlineExpert(
                known_item_mask=(
                    bpr.known_item_mask_
                ),
                popularity_scores=(
                    popularity.item_scores_
                ),
                decay=online_decay,
                prior_strength=float(
                    window_size
                ),
            )
        )

        methods = {
            spec["name"]:
                build_method(spec)
            for spec in method_specs
        }

        per_user_method_loss = {
            spec["name"]: 0.0
            for spec in method_specs
        }

        historical_total_loss = 0.0
        online_total_loss = 0.0
        valid_window_count = 0

        for window in windows:
            known_counts = (
                filter_counts_by_known_mask(
                    window.item_counts,
                    bpr.known_item_mask_,
                )
            )

            if not known_counts:
                continue

            # ---------------------------------------------
            # SHIFT SIGNAL AVAILABLE BEFORE THIS WINDOW
            # ---------------------------------------------

            online_profile = (
                online_expert
                .observed_profile()
            )

            if online_profile:
                shift_signal = (
                    jensen_shannon_from_counts(
                        historical_counts,
                        online_profile,
                    )
                )
            else:
                shift_signal = 0.0

            # ---------------------------------------------
            # EXPERT PREDICTIONS AVAILABLE BEFORE OBSERVING
            # CURRENT WINDOW
            # ---------------------------------------------

            online_recommendations = (
                online_expert.recommend(
                    k=k
                )
            )

            historical_coverage = (
                event_mass_at_k(
                    historical_recommendations,
                    known_counts,
                    k=k,
                )
            )

            online_coverage = (
                event_mass_at_k(
                    online_recommendations,
                    known_counts,
                    k=k,
                )
            )

            historical_loss = (
                1.0
                - historical_coverage
            )

            online_loss = (
                1.0
                - online_coverage
            )

            historical_total_loss += (
                historical_loss
            )

            online_total_loss += (
                online_loss
            )

            valid_window_count += 1

            # ---------------------------------------------
            # META-ALGORITHMS
            # ---------------------------------------------

            for spec in method_specs:
                name = spec["name"]

                method = methods[name]

                alpha = (
                    method.historical_weight(
                        shift_signal
                    )
                )

                method_loss = (
                    expected_mixture_loss(
                        alpha,
                        historical_loss,
                        online_loss,
                    )
                )

                per_user_method_loss[
                    name
                ] += method_loss

                window_rows.append(
                    {
                        "user_idx":
                            user_idx,
                        "window_number":
                            window.window_index
                            + 1,
                        "method":
                            name,
                        "family":
                            spec["family"],
                        "historical_weight":
                            alpha,
                        "shift_signal":
                            shift_signal,
                        "historical_loss":
                            historical_loss,
                        "online_loss":
                            online_loss,
                        "method_loss":
                            method_loss,
                        "historical_coverage":
                            historical_coverage,
                        "online_coverage":
                            online_coverage,
                        "expected_coverage":
                            1.0
                            - method_loss,
                    }
                )

                # Update AFTER evaluating the current window.
                method.observe(
                    historical_loss,
                    online_loss,
                )

            # ---------------------------------------------
            # ONLINE EXPERT UPDATE AFTER CURRENT WINDOW
            # ---------------------------------------------

            online_expert.update(
                known_counts
            )

        if valid_window_count == 0:
            continue

        best_fixed_expert_loss = min(
            historical_total_loss,
            online_total_loss,
        )

        for spec in method_specs:
            name = spec["name"]

            total_loss = (
                per_user_method_loss[
                    name
                ]
            )

            user_rows.append(
                {
                    "user_idx":
                        user_idx,
                    "method":
                        name,
                    "family":
                        spec["family"],
                    "n_windows":
                        valid_window_count,
                    "total_loss":
                        total_loss,
                    "mean_loss":
                        total_loss
                        / valid_window_count,
                    "expected_coverage":
                        1.0
                        - (
                            total_loss
                            / valid_window_count
                        ),
                    "historical_total_loss":
                        historical_total_loss,
                    "online_total_loss":
                        online_total_loss,
                    "best_fixed_expert_loss":
                        best_fixed_expert_loss,
                    "regret_to_best_fixed_expert":
                        total_loss
                        - best_fixed_expert_loss,
                }
            )

        analyzed_users += 1

        if (
            analyzed_users % 100
            == 0
        ):
            print(
                f"  analyzed "
                f"{analyzed_users:,} users..."
            )

    print(
        f"Users analyzed: "
        f"{analyzed_users:,}"
    )

    print(
        f"Users skipped (<{min_windows} "
        f"complete windows): "
        f"{skipped_users:,}"
    )

    if not window_rows:
        raise RuntimeError(
            "No valid sequential windows were generated."
        )

    return (
        pd.DataFrame(
            window_rows
        ),
        pd.DataFrame(
            user_rows
        ),
    )


def summarize_methods(
    per_user: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        per_user
        .groupby(
            [
                "method",
                "family",
            ],
            as_index=False,
        )
        .agg(
            users=(
                "user_idx",
                "nunique",
            ),
            mean_user_loss=(
                "mean_loss",
                "mean",
            ),
            std_user_loss=(
                "mean_loss",
                "std",
            ),
            mean_expected_coverage=(
                "expected_coverage",
                "mean",
            ),
            mean_regret=(
                "regret_to_best_fixed_expert",
                "mean",
            ),
            median_regret=(
                "regret_to_best_fixed_expert",
                "median",
            ),
        )
    )

    summary[
        "sem_user_loss"
    ] = (
        summary[
            "std_user_loss"
        ]
        / np.sqrt(
            summary[
                "users"
            ]
        )
    )

    return summary


def select_validation_parameters(
    validation_summary: pd.DataFrame,
    method_specs: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    spec_by_name = {
        spec["name"]: spec
        for spec in method_specs
    }

    selected: dict[
        str,
        Any,
    ] = {}

    for family in [
        "static",
        "hedge",
        "shift",
    ]:
        family_results = (
            validation_summary[
                validation_summary[
                    "family"
                ]
                == family
            ]
            .sort_values(
                "mean_user_loss",
                ascending=True,
            )
        )

        if family_results.empty:
            raise RuntimeError(
                f"No validation results "
                f"for family {family}"
            )

        best_name = str(
            family_results.iloc[0][
                "method"
            ]
        )

        selected[family] = (
            spec_by_name[
                best_name
            ]["params"]
        )

        selected[
            f"{family}_selected_name"
        ] = best_name

        selected[
            f"{family}_validation_loss"
        ] = float(
            family_results.iloc[0][
                "mean_user_loss"
            ]
        )

    return selected


def add_display_names(
    dataframe: pd.DataFrame,
    specs: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    names = {
        spec["name"]:
            spec.get(
                "display_name",
                spec["name"],
            )
        for spec in specs
    }

    output = dataframe.copy()

    output[
        "display_name"
    ] = output[
        "method"
    ].map(names)

    return output


def create_figures(
    per_window: pd.DataFrame,
    per_user: pd.DataFrame,
    summary: pd.DataFrame,
    specs: list[
        dict[str, Any]
    ],
    k: int,
) -> None:
    sns.set_theme(
        style="whitegrid"
    )

    summary = add_display_names(
        summary,
        specs,
    )

    per_window = add_display_names(
        per_window,
        specs,
    )

    per_user = add_display_names(
        per_user,
        specs,
    )

    order = [
        spec.get(
            "display_name",
            spec["name"],
        )
        for spec in specs
    ]

    # ========================================================
    # FIGURE 1: FINAL EXPECTED COVERAGE
    # ========================================================

    plot_summary = (
        summary
        .set_index(
            "display_name"
        )
        .loc[order]
        .reset_index()
    )

    coverage_sem = (
        plot_summary[
            "sem_user_loss"
        ]
        .to_numpy()
    )

    plt.figure(
        figsize=(10, 5)
    )

    x = np.arange(
        len(plot_summary)
    )

    plt.bar(
        x,
        plot_summary[
            "mean_expected_coverage"
        ],
        yerr=(
            1.96
            * coverage_sem
        ),
        capsize=4,
    )

    plt.xticks(
        x,
        plot_summary[
            "display_name"
        ],
        rotation=15,
        ha="right",
    )

    plt.ylabel(
        f"Expected Event Coverage@{k}"
    )

    plt.title(
        "Robust Prediction Combination "
        "on the Held-Out Test Stream"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "robust_learning_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 2: CUMULATIVE MEAN LOSS
    # ========================================================

    window_summary = (
        per_window
        .groupby(
            [
                "display_name",
                "window_number",
            ],
            as_index=False,
        )
        .agg(
            mean_loss=(
                "method_loss",
                "mean",
            ),
            users=(
                "user_idx",
                "nunique",
            ),
        )
    )

    window_summary = (
        window_summary[
            window_summary[
                "users"
            ]
            >= 20
        ]
        .copy()
    )

    window_summary[
        "cumulative_mean_loss"
    ] = (
        window_summary
        .sort_values(
            "window_number"
        )
        .groupby(
            "display_name"
        )["mean_loss"]
        .cumsum()
    )

    plt.figure(
        figsize=(10, 6)
    )

    for name in order:
        method_df = (
            window_summary[
                window_summary[
                    "display_name"
                ]
                == name
            ]
            .sort_values(
                "window_number"
            )
        )

        if method_df.empty:
            continue

        plt.plot(
            method_df[
                "window_number"
            ],
            method_df[
                "cumulative_mean_loss"
            ],
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Equal-interaction test window"
    )

    plt.ylabel(
        "Cumulative mean loss"
    )

    plt.title(
        "Cumulative Sequential Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "cumulative_loss.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 3: TRUST OVER TIME
    # ========================================================

    trust_names = [
        "Static Mixture",
        "Hedge",
        "Shift-Aware Trust",
    ]

    trust_data = (
        per_window[
            per_window[
                "display_name"
            ].isin(
                trust_names
            )
        ]
        .groupby(
            [
                "display_name",
                "window_number",
            ],
            as_index=False,
        )
        .agg(
            mean_weight=(
                "historical_weight",
                "mean",
            ),
            std_weight=(
                "historical_weight",
                "std",
            ),
            users=(
                "user_idx",
                "nunique",
            ),
        )
    )

    trust_data = (
        trust_data[
            trust_data["users"]
            >= 20
        ]
        .copy()
    )

    trust_data[
        "sem"
    ] = (
        trust_data[
            "std_weight"
        ]
        / np.sqrt(
            trust_data[
                "users"
            ]
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    for name in trust_names:
        method_df = (
            trust_data[
                trust_data[
                    "display_name"
                ]
                == name
            ]
            .sort_values(
                "window_number"
            )
        )

        if method_df.empty:
            continue

        x_values = method_df[
            "window_number"
        ].to_numpy()

        means = method_df[
            "mean_weight"
        ].to_numpy()

        ci = (
            1.96
            * method_df[
                "sem"
            ].to_numpy()
        )

        plt.plot(
            x_values,
            means,
            marker="o",
            label=name,
        )

        plt.fill_between(
            x_values,
            means - ci,
            means + ci,
            alpha=0.15,
        )

    plt.xlabel(
        "Equal-interaction test window"
    )

    plt.ylabel(
        "Weight on historical BPR predictor"
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.title(
        "Evolution of Historical Prediction Trust"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "trust_over_time.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FIGURE 4: PERFORMANCE BY SHIFT
    # ========================================================

    unique_windows = (
        per_window[
            [
                "user_idx",
                "window_number",
                "shift_signal",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    unique_windows[
        "shift_bin"
    ] = pd.qcut(
        unique_windows[
            "shift_signal"
        ],
        q=5,
        duplicates="drop",
    )

    bin_map = (
        unique_windows[
            [
                "user_idx",
                "window_number",
                "shift_bin",
            ]
        ]
    )

    shift_plot = (
        per_window.merge(
            bin_map,
            on=[
                "user_idx",
                "window_number",
            ],
            how="left",
        )
    )

    selected_names = [
        "Prediction Only",
        "Online Only",
        "Hedge",
        "Shift-Aware Trust",
    ]

    shift_plot = (
        shift_plot[
            shift_plot[
                "display_name"
            ].isin(
                selected_names
            )
        ]
    )

    grouped = (
        shift_plot
        .groupby(
            [
                "display_name",
                "shift_bin",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            mean_coverage=(
                "expected_coverage",
                "mean",
            ),
            mean_shift=(
                "shift_signal",
                "mean",
            ),
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    for name in selected_names:
        method_df = (
            grouped[
                grouped[
                    "display_name"
                ]
                == name
            ]
            .sort_values(
                "mean_shift"
            )
        )

        if method_df.empty:
            continue

        plt.plot(
            method_df[
                "mean_shift"
            ],
            method_df[
                "mean_coverage"
            ],
            marker="o",
            linewidth=2,
            label=name,
        )

    plt.xlabel(
        "Behavioral shift "
        "(Jensen-Shannon divergence)"
    )

    plt.ylabel(
        f"Expected Event Coverage@{k}"
    )

    plt.title(
        "Robustness as Behavioral Shift Increases"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "performance_by_shift.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 4: robust combination "
            "of historical and online predictors."
        )
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
    )

    parser.add_argument(
        "--min-windows",
        type=int,
        default=DEFAULT_MIN_WINDOWS,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
    )

    parser.add_argument(
        "--online-decay",
        type=float,
        default=DEFAULT_ONLINE_DECAY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    args = parser.parse_args()

    if args.window_size <= 0:
        raise ValueError(
            "--window-size must be positive."
        )

    if args.min_windows < 2:
        raise ValueError(
            "--min-windows must be >= 2."
        )

    if args.k <= 0:
        raise ValueError(
            "--k must be positive."
        )

    if not (
        0.0
        <= args.online_decay
        < 1.0
    ):
        raise ValueError(
            "--online-decay must lie in [0, 1)."
        )

    set_seed(
        args.seed
    )

    ensure_dir(
        TABLE_DIR
    )

    ensure_dir(
        FIGURE_DIR
    )

    print("=" * 72)
    print(
        "MILESTONE 4: ROBUST PREDICTION LEARNING"
    )
    print("=" * 72)

    print(
        f"Mode: "
        f"{'QUICK' if args.quick else 'FULL'}"
    )

    print(
        f"Window size: "
        f"{args.window_size}"
    )

    print(
        f"Online decay: "
        f"{args.online_decay}"
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

    test = load_interactions(
        TEST_FILE
    )

    bpr_parameters = (
        load_bpr_parameters(
            quick=args.quick
        )
    )

    print()
    print(
        "BPR parameters"
    )
    print("-" * 40)

    print(
        json.dumps(
            bpr_parameters,
            indent=2,
        )
    )

    # ========================================================
    # VALIDATION PARAMETER SELECTION
    # ========================================================

    print()
    print("=" * 72)
    print(
        "VALIDATION: ROBUST-LEARNING PARAMETER SELECTION"
    )
    print("=" * 72)

    validation_bpr, validation_popularity = (
        fit_historical_models(
            history=train,
            bpr_parameters=(
                bpr_parameters
            ),
            seed=args.seed,
        )
    )

    validation_specs = (
        validation_method_specs()
    )

    validation_max_users = (
        100
        if args.quick
        else None
    )

    (
        validation_windows,
        validation_users,
    ) = evaluate_sequential_stream(
        history=train,
        stream=validation,
        bpr=validation_bpr,
        popularity=validation_popularity,
        method_specs=validation_specs,
        window_size=args.window_size,
        min_windows=args.min_windows,
        k=args.k,
        online_decay=(
            args.online_decay
        ),
        max_users=(
            validation_max_users
        ),
        seed=args.seed,
    )

    validation_summary = (
        summarize_methods(
            validation_users
        )
    )

    validation_summary = (
        validation_summary.sort_values(
            "mean_user_loss"
        )
    )

    print()
    print(
        validation_summary[
            [
                "method",
                "family",
                "users",
                "mean_user_loss",
                "mean_expected_coverage",
                "mean_regret",
            ]
        ].to_string(
            index=False
        )
    )

    validation_summary_path = (
        TABLE_DIR
        / "robust_learning_validation.csv"
    )

    validation_summary.to_csv(
        validation_summary_path,
        index=False,
    )

    selected_parameters = (
        select_validation_parameters(
            validation_summary,
            validation_specs,
        )
    )

    selected_parameters[
        "window_size"
    ] = args.window_size

    selected_parameters[
        "online_decay"
    ] = args.online_decay

    selected_parameters[
        "ranking_k"
    ] = args.k

    selected_path = (
        TABLE_DIR
        / "selected_robust_parameters.json"
    )

    save_json(
        selected_parameters,
        selected_path,
    )

    print()
    print(
        "Selected robust-learning parameters"
    )

    print("-" * 40)

    print(
        json.dumps(
            selected_parameters,
            indent=2,
        )
    )

    # ========================================================
    # FINAL TEST HISTORY = TRAIN + VALIDATION
    # ========================================================

    print()
    print("=" * 72)
    print(
        "FINAL TEST: FIT HISTORICAL MODELS "
        "ON TRAIN + VALIDATION"
    )
    print("=" * 72)

    train_validation = (
        pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "user_idx",
                "timestamp",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    final_bpr, final_popularity = (
        fit_historical_models(
            history=train_validation,
            bpr_parameters=(
                bpr_parameters
            ),
            seed=args.seed,
        )
    )

    test_specs = (
        selected_test_specs(
            selected_parameters
        )
    )

    test_max_users = (
        100
        if args.quick
        else None
    )

    (
        test_windows,
        test_users,
    ) = evaluate_sequential_stream(
        history=train_validation,
        stream=test,
        bpr=final_bpr,
        popularity=final_popularity,
        method_specs=test_specs,
        window_size=args.window_size,
        min_windows=args.min_windows,
        k=args.k,
        online_decay=(
            args.online_decay
        ),
        max_users=test_max_users,
        seed=args.seed,
    )

    test_summary = (
        summarize_methods(
            test_users
        )
    )

    # Add human-readable names.
    display_map = {
        spec["name"]:
            spec.get(
                "display_name",
                spec["name"],
            )
        for spec in test_specs
    }

    test_summary[
        "display_name"
    ] = (
        test_summary[
            "method"
        ].map(
            display_map
        )
    )

    test_summary = (
        test_summary.sort_values(
            "mean_user_loss"
        )
    )

    print()
    print("=" * 72)
    print(
        "FINAL TEST RESULTS"
    )
    print("=" * 72)

    print(
        test_summary[
            [
                "display_name",
                "users",
                "mean_user_loss",
                "mean_expected_coverage",
                "mean_regret",
                "median_regret",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    test_summary_path = (
        TABLE_DIR
        / "robust_learning_test.csv"
    )

    per_user_path = (
        TABLE_DIR
        / "robust_learning_per_user.csv"
    )

    per_window_path = (
        TABLE_DIR
        / "robust_learning_windows.csv"
    )

    test_summary.to_csv(
        test_summary_path,
        index=False,
    )

    test_users.to_csv(
        per_user_path,
        index=False,
    )

    test_windows.to_csv(
        per_window_path,
        index=False,
    )

    # ========================================================
    # FIGURES
    # ========================================================

    create_figures(
        per_window=test_windows,
        per_user=test_users,
        summary=test_summary,
        specs=test_specs,
        k=args.k,
    )

    print()
    print("=" * 72)
    print(
        "GENERATED OUTPUTS"
    )
    print("=" * 72)

    print(
        validation_summary_path
    )

    print(
        selected_path
    )

    print(
        test_summary_path
    )

    print(
        per_user_path
    )

    print(
        per_window_path
    )

    print(
        FIGURE_DIR
        / "robust_learning_comparison.png"
    )

    print(
        FIGURE_DIR
        / "cumulative_loss.png"
    )

    print(
        FIGURE_DIR
        / "trust_over_time.png"
    )

    print(
        FIGURE_DIR
        / "performance_by_shift.png"
    )


if __name__ == "__main__":
    main()
