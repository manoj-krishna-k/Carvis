"""
CARVIS - Train Model on Real Data
--------------------------------------
Picks ONE driver from the dataset to be "the owner" of our fictional car.
Trains Isolation Forest ONLY on that driver's "Normal" behavior windows.

Then tests:
1. The SAME owner's OTHER trips (never seen during training)        -> should mostly say NORMAL
2. ALL OTHER drivers' trips                                          -> should mostly say ANOMALY
3. The owner's own "Aggressive" trips (if they have any)              -> interesting edge case to discuss

This mirrors exactly how CARVIS would work in real life: the model
only ever learns ONE person's driving, then flags anyone else.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

# CHANGE THIS to whichever driver you want to be "the owner" for your demo
OWNER_DRIVER_ID = "D1"

FEATURE_COLUMNS = [
    "accel_x_mean", "accel_x_std", "accel_y_mean", "accel_y_std", "accel_z_std",
    "yaw_std", "yaw_range", "harsh_accel_count", "harsh_turn_count",
    "speed_mean", "speed_std"
]

if __name__ == "__main__":
    df = pd.read_csv("../data/all_features.csv")

    # Drop rows with missing speed data (in case some trips lack RAW_GPS.txt)
    df = df.dropna(subset=FEATURE_COLUMNS)

    # ---- Split: Owner's NORMAL behavior windows = training data ----
    owner_normal = df[(df["driver_id"] == OWNER_DRIVER_ID) & (df["behavior"] == "Normal")]

    if len(owner_normal) == 0:
        print(f"ERROR: No 'Normal' behavior windows found for driver {OWNER_DRIVER_ID}")
        print("Available drivers/behaviors:")
        print(df.groupby(["driver_id", "behavior"]).size())
        exit()

    print(f"Training on owner ({OWNER_DRIVER_ID}) Normal driving: {len(owner_normal)} windows")

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,   # assume max 5% noisy/unusual behavior even for the owner
        random_state=42
    )
    model.fit(owner_normal[FEATURE_COLUMNS])

    # ---- Save model + metadata ----
    os.makedirs("../trained_models", exist_ok=True)
    joblib.dump(model, "../trained_models/isolation_forest.pkl")
    joblib.dump(FEATURE_COLUMNS, "../trained_models/feature_columns.pkl")
    joblib.dump(OWNER_DRIVER_ID, "../trained_models/owner_driver_id.pkl")

    print(f"Model saved to ../trained_models/isolation_forest.pkl")

    # ---- Quick sanity check: test on everyone ----
    print(f"\n{'=' * 60}")
    print("QUICK SANITY CHECK (full evaluation happens in evaluate_model.py)")
    print(f"{'=' * 60}")

    for driver in sorted(df["driver_id"].unique()):
        driver_data = df[df["driver_id"] == driver]
        preds = model.predict(driver_data[FEATURE_COLUMNS])
        normal_pct = (preds == 1).sum() / len(preds) * 100
        tag = "<- OWNER" if driver == OWNER_DRIVER_ID else ""
        print(f"  {driver}: {normal_pct:.1f}% predicted as NORMAL  {tag}")
