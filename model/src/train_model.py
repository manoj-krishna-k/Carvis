"""
CARVIS - Train Model (v3: calibrated XGBoost)
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from extract_features import FEATURE_COLUMNS
from data_split import split_by_trip, make_binary_labels

OWNER_DRIVER: str = "D1"
TRAIN_FRACTION: float = 0.7
MODEL_DIR = "../trained_models"

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=4,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.1,
    random_state=42,
    eval_metric="logloss",
)


def load_features(path: str = "../data/all_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"all_features.csv is missing expected feature columns: {missing}. "
            "Run extract_features.py after updating load_real_data.py."
        )
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    dropped = len(df) - len(clean)
    if dropped > 0:
        print(f"Dropped {dropped} window(s) containing NaN/inf feature values.")
    return clean.reset_index(drop=True)


def main() -> None:
    df = load_features()
    df = drop_invalid_rows(df)

    n_drivers = df["driver_id"].nunique()
    print(f"Loaded {len(df)} total windows across {n_drivers} drivers.")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")

    if n_drivers < 2:
        raise ValueError("Need at least 2 drivers to train owner-vs-intruder model.")

    split = split_by_trip(df, train_fraction=TRAIN_FRACTION)

    print("\nTrip split by driver:")
    for driver_id in sorted(split.train_trips_by_driver.keys()):
        n_train_trips = len(split.train_trips_by_driver[driver_id])
        n_test_trips = len(split.test_trips_by_driver[driver_id])
        tag = " <- OWNER" if driver_id == OWNER_DRIVER else ""
        print(f"  {driver_id}: {n_train_trips} train trip(s), {n_test_trips} test trip(s){tag}")

    X_train = split.train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = make_binary_labels(split.train, OWNER_DRIVER)

    print(
        f"\nTrain set: {len(split.train)} windows "
        f"({int(y_train.sum())} owner / {int((1 - y_train).sum())} other)"
    )
    print(f"Test set:  {len(split.test)} windows")

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set ended up with only one class present.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    n_negative = int((1 - y_train).sum())
    n_positive = int(y_train.sum())
    # Mild reweighting keeps probabilities better calibrated than the full ratio.
    scale_pos_weight = float(np.sqrt(n_negative / n_positive)) if n_positive > 0 else 1.0

    print(
        f"\nClass balance in training: {n_positive} owner / {n_negative} other "
        f"-> scale_pos_weight = {scale_pos_weight:.2f}"
    )

    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train_scaled, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    joblib.dump(OWNER_DRIVER, os.path.join(MODEL_DIR, "owner_driver.pkl"))
    split.test.to_csv(os.path.join(MODEL_DIR, "test_split.csv"), index=False)

    print(f"\nSaved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
