"""
CARVIS - Trip-Based Train/Test Split
-------------------------------------------
Splits feature windows into train/test sets by TRIP, never by individual
window. This prevents data leakage: windows from the same trip are highly
correlated (same drive, same road, similar conditions), so letting some
windows from a trip land in train and others in test would let the model
"cheat" by recognizing the trip rather than the underlying driving
behavior.

Split design (applies to EVERY driver, not just the owner):
    For each driver (owner and all others), their trips are independently
    split into a train portion and a test portion. This guarantees:
      - XGBoost sees BOTH classes (owner=1, other drivers=0) during training
      - No trip ever appears in both train and test
      - Test set still contains unseen owner trips AND unseen other-driver
        trips, so recall and precision are both measured honestly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TripSplit:
    """Holds the train/test feature DataFrames after a trip-based split."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_trips_by_driver: dict[str, list[str]]
    test_trips_by_driver: dict[str, list[str]]


def split_by_trip(
    features_df: pd.DataFrame,
    train_fraction: float = 0.7,
    random_state: int = 42,
) -> TripSplit:
    """
    Performs a per-driver, trip-based split. Every driver present in
    `features_df` has their own trips shuffled and split independently,
    so the train/test ratio is consistent across drivers regardless of
    how many trips each one has.

    Args:
        features_df: output of extract_features_per_trip(), must contain
            columns driver_id and trip_folder.
        train_fraction: fraction of EACH driver's trips used for training.
        random_state: seed for reproducible trip selection.

    Returns:
        TripSplit with .train and .test DataFrames, plus a record of
        exactly which trips went where, for auditability.
    """
    rng = np.random.RandomState(random_state)

    train_trips_by_driver: dict[str, list[str]] = {}
    test_trips_by_driver: dict[str, list[str]] = {}

    train_frames = []
    test_frames = []

    for driver_id, driver_df in features_df.groupby("driver_id"):
        trips = sorted(driver_df["trip_folder"].unique())
        shuffled = trips.copy()
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            # Only one trip for this driver -- put it in train AND warn,
            # since we can't honestly hold out a test trip for them.
            n_train = 1
        else:
            n_train = max(1, int(round(len(shuffled) * train_fraction)))
            n_train = min(n_train, len(shuffled) - 1)  # always leave >=1 for test

        train_trips = sorted(shuffled[:n_train])
        test_trips = sorted(shuffled[n_train:])

        train_trips_by_driver[driver_id] = train_trips
        test_trips_by_driver[driver_id] = test_trips

        train_frames.append(driver_df[driver_df["trip_folder"].isin(train_trips)])
        if test_trips:
            test_frames.append(driver_df[driver_df["trip_folder"].isin(test_trips)])

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    return TripSplit(
        train=train_df,
        test=test_df,
        train_trips_by_driver=train_trips_by_driver,
        test_trips_by_driver=test_trips_by_driver,
    )


def make_binary_labels(df: pd.DataFrame, owner_driver: str) -> np.ndarray:
    """Owner windows -> 1, every other driver's windows -> 0."""
    return (df["driver_id"] == owner_driver).astype(int).to_numpy()
