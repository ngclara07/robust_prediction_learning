from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LASTFM_COLUMNS = [
    "user_id",
    "timestamp",
    "artist_id",
    "artist_name",
    "track_id",
    "track_name",
]


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_lastfm(path: str | Path) -> pd.DataFrame:
    """
    Load the Last.fm-1K listening-event dataset.

    Expected schema:
        user_id, timestamp, artist_id, artist_name, track_id, track_name
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        names=LASTFM_COLUMNS,
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )

    return df


def preprocess_interactions(
    df: pd.DataFrame,
    min_user_interactions: int = 100,
    min_artist_interactions: int = 20,
) -> pd.DataFrame:
    """
    Clean Last.fm interactions and prepare an artist-level event dataset.

    The study uses artist-level interactions to reduce extreme track sparsity.
    """

    required = {
        "user_id",
        "timestamp",
        "artist_name",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()

    # Remove unusable records.
    data = data.dropna(
        subset=["user_id", "timestamp", "artist_name"]
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    data = data.dropna(subset=["timestamp"])

    # Normalize strings.
    data["user_id"] = data["user_id"].astype(str).str.strip()
    data["artist_name"] = data["artist_name"].astype(str).str.strip()

    data = data[
        (data["user_id"] != "")
        & (data["artist_name"] != "")
    ]

    # Artist MBIDs are not always available in Last.fm-1K.
    # Use normalized artist names as the canonical item identifier.
    data["artist_key"] = (
        data["artist_name"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Filter extremely sparse users.
    user_counts = data["user_id"].value_counts()
    valid_users = user_counts[
        user_counts >= min_user_interactions
    ].index

    data = data[data["user_id"].isin(valid_users)]

    # Filter extremely rare artists.
    artist_counts = data["artist_key"].value_counts()
    valid_artists = artist_counts[
        artist_counts >= min_artist_interactions
    ].index

    data = data[data["artist_key"].isin(valid_artists)]

    # Re-check users because artist filtering may reduce their histories.
    user_counts = data["user_id"].value_counts()
    valid_users = user_counts[
        user_counts >= min_user_interactions
    ].index

    data = data[data["user_id"].isin(valid_users)]

    # Encode IDs deterministically.
    user_values = sorted(data["user_id"].unique())
    artist_values = sorted(data["artist_key"].unique())

    user_mapping = {
        value: idx
        for idx, value in enumerate(user_values)
    }

    artist_mapping = {
        value: idx
        for idx, value in enumerate(artist_values)
    }

    data["user_idx"] = data["user_id"].map(user_mapping).astype("int32")
    data["artist_idx"] = (
        data["artist_key"].map(artist_mapping).astype("int32")
    )

    data = data.sort_values(
        ["user_idx", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    return data[
        [
            "user_id",
            "user_idx",
            "artist_name",
            "artist_key",
            "artist_idx",
            "timestamp",
        ]
    ]


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> TemporalSplit:
    """
    Perform a per-user chronological train/validation/test split.

    Default:
        70% train
        10% validation
        20% test

    No interactions from the future are allowed into a user's training history.
    """

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be < 1."
        )

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_df in df.groupby("user_idx", sort=False):
        user_df = user_df.sort_values("timestamp")

        n = len(user_df)

        train_end = int(n * train_fraction)
        validation_end = int(
            n * (train_fraction + validation_fraction)
        )

        # Guarantee non-empty partitions where possible.
        train_end = max(train_end, 1)
        validation_end = max(validation_end, train_end + 1)

        if validation_end >= n:
            continue

        train_parts.append(user_df.iloc[:train_end])
        validation_parts.append(
            user_df.iloc[train_end:validation_end]
        )
        test_parts.append(user_df.iloc[validation_end:])

    if not train_parts:
        raise ValueError("No users remain after temporal splitting.")

    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )


def validate_temporal_split(split: TemporalSplit) -> None:
    """
    Verify that every user's train data precedes validation,
    and validation precedes test.
    """

    users = set(split.train["user_idx"])

    for user_idx in users:
        train_u = split.train[
            split.train["user_idx"] == user_idx
        ]

        val_u = split.validation[
            split.validation["user_idx"] == user_idx
        ]

        test_u = split.test[
            split.test["user_idx"] == user_idx
        ]

        if val_u.empty or test_u.empty:
            continue

        if train_u["timestamp"].max() > val_u["timestamp"].min():
            raise AssertionError(
                f"Train/validation leakage for user {user_idx}"
            )

        if val_u["timestamp"].max() > test_u["timestamp"].min():
            raise AssertionError(
                f"Validation/test leakage for user {user_idx}"
            )


def save_processed_data(
    df: pd.DataFrame,
    split: TemporalSplit,
    output_dir: str | Path,
) -> None:
    """Save processed interactions and temporal partitions as Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        output_dir / "interactions.parquet",
        index=False,
    )

    split.train.to_parquet(
        output_dir / "train.parquet",
        index=False,
    )

    split.validation.to_parquet(
        output_dir / "validation.parquet",
        index=False,
    )

    split.test.to_parquet(
        output_dir / "test.parquet",
        index=False,
    )
