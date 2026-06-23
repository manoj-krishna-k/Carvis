"""
CARVIS - Export Test Pool for Backend
-------------------------------------------
Run this AFTER train_model.py and evaluate_model.py work correctly.

This creates two files that the FastAPI backend reads directly:
    1. test_pool.csv    -> every feature window + its true driver/behavior label
                           (used for the "Test Random Driver" button)
    2. model_stats.csv  -> precision, recall, accuracy numbers
                           (used for the "Model Performance" screen)

Also COPIES the trained model files into backend/trained_models/ so the
backend has everything it needs in one place.
"""

import pandas as pd
import joblib
import shutil
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

FEATURE_COLUMNS = [
    "accel_x_mean", "accel_x_std", "accel_y_mean", "accel_y_std", "accel_z_std",
    "yaw_std", "yaw_range", "harsh_accel_count", "harsh_turn_count",
    "speed_mean", "speed_std"
]

if __name__ == "__main__":
    model = joblib.load("../trained_models/isolation_forest.pkl")
    OWNER_DRIVER_ID = joblib.load("../trained_models/owner_driver_id.pkl")

    df = pd.read_csv("../data/all_features.csv")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    # ---- Save the full feature set as the test pool (with ground truth) ----
    test_pool = df[FEATURE_COLUMNS + ["driver_id", "behavior", "road", "trip_folder"]].copy()
    test_pool.to_csv("../trained_models/test_pool.csv", index=False)
    print(f"Saved test_pool.csv with {len(test_pool)} windows")

    # ---- Compute and save model performance stats ----
    df["true_label"] = df["driver_id"].apply(lambda d: 1 if d == OWNER_DRIVER_ID else -1)
    predictions = model.predict(df[FEATURE_COLUMNS])

    stats = {
        "owner_driver_id": OWNER_DRIVER_ID,
        "total_windows_tested": len(df),
        "precision": round(precision_score(df["true_label"], predictions, pos_label=1), 3),
        "recall": round(recall_score(df["true_label"], predictions, pos_label=1), 3),
        "accuracy": round(accuracy_score(df["true_label"], predictions), 3),
        "f1_score": round(f1_score(df["true_label"], predictions, pos_label=1), 3),
        "num_drivers_tested": df["driver_id"].nunique(),
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv("../trained_models/model_stats.csv", index=False)
    print(f"Saved model_stats.csv:")
    print(stats_df.T)

    # ---- Copy everything backend needs into backend/trained_models/ ----
    backend_models_dir = "../../backend/trained_models"
    os.makedirs(backend_models_dir, exist_ok=True)

    files_to_copy = [
        "isolation_forest.pkl",
        "feature_columns.pkl",
        "owner_driver_id.pkl",
        "test_pool.csv",
        "model_stats.csv",
    ]

    for filename in files_to_copy:
        src = os.path.join("../trained_models", filename)
        dst = os.path.join(backend_models_dir, filename)
        shutil.copy(src, dst)

    print(f"\nCopied all model files to: backend/trained_models/")
    print("Backend is ready to run.")
