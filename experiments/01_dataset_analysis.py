from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robust_prediction_learning.data import (
    load_lastfm,
    preprocess_interactions,
    save_processed_data,
    temporal_split,
    validate_temporal_split,
)
from robust_prediction_learning.utils import ensure_dir, save_json


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "lastfm-dataset-1K"
    / "userid-timestamp-artid-artname-traid-traname.tsv"
)

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"


def main() -> None:
    ensure_dir(PROCESSED_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(TABLE_DIR)

    print("Loading Last.fm-1K...")
    raw = load_lastfm(RAW_FILE)

    print(f"Raw interactions: {len(raw):,}")

    print("Preprocessing...")
    interactions = preprocess_interactions(raw)

    print("Constructing chronological splits...")
    split = temporal_split(interactions)

    validate_temporal_split(split)

    print("Temporal leakage check: PASSED")

    save_processed_data(
        interactions,
        split,
        PROCESSED_DIR,
    )

    stats = {
        "raw_interactions": len(raw),
        "processed_interactions": len(interactions),
        "users": interactions["user_idx"].nunique(),
        "artists": interactions["artist_idx"].nunique(),
        "earliest_timestamp": interactions["timestamp"].min(),
        "latest_timestamp": interactions["timestamp"].max(),
        "train_interactions": len(split.train),
        "validation_interactions": len(split.validation),
        "test_interactions": len(split.test),
    }

    save_json(
        stats,
        TABLE_DIR / "dataset_statistics.json",
    )

    print("\nDataset summary")
    print("-" * 50)

    for key, value in stats.items():
        print(f"{key:25s}: {value}")

    # --------------------------------------------------
    # Figure 1: interactions per user
    # --------------------------------------------------

    user_counts = (
        interactions.groupby("user_idx")
        .size()
        .rename("interactions")
    )

    plt.figure(figsize=(8, 5))

    sns.histplot(
        user_counts,
        bins=50,
        log_scale=(True, False),
    )

    plt.xlabel("Interactions per user")
    plt.ylabel("Number of users")
    plt.title("Distribution of Listening Activity")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "interactions_per_user.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------
    # Figure 2: artist popularity
    # --------------------------------------------------

    artist_counts = (
        interactions.groupby("artist_idx")
        .size()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(8, 5))

    plt.loglog(
        range(1, len(artist_counts) + 1),
        artist_counts.values,
    )

    plt.xlabel("Artist popularity rank")
    plt.ylabel("Number of interactions")
    plt.title("Artist Popularity Distribution")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "artist_popularity.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------
    # Figure 3: interactions over time
    # --------------------------------------------------

    monthly = (
        interactions
        .set_index("timestamp")
        .resample("ME")
        .size()
    )

    plt.figure(figsize=(10, 5))

    monthly.plot()

    plt.xlabel("Date")
    plt.ylabel("Interactions")
    plt.title("Last.fm Listening Events Over Time")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "interactions_over_time.png",
        dpi=200,
    )

    plt.close()

    print("\nGenerated:")
    print(PROCESSED_DIR / "interactions.parquet")
    print(PROCESSED_DIR / "train.parquet")
    print(PROCESSED_DIR / "validation.parquet")
    print(PROCESSED_DIR / "test.parquet")
    print(TABLE_DIR / "dataset_statistics.json")


if __name__ == "__main__":
    main()
