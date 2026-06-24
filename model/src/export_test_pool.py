"""
CARVIS - Export Test Pool for Backend
-------------------------------------------
Run this AFTER train_model.py and evaluate_model.py succeed.

Copies the trained XGBoost artifacts + the held-out test split + model
stats into backend/trained_models/, so the FastAPI backend has everything
it needs without any manual file copying.
"""

from __future__ import annotations

import os
import shutil

FILES_TO_COPY = [
    "xgb_model.pkl",
    "scaler.pkl",
    "feature_columns.pkl",
    "owner_driver.pkl",
    "test_split.csv",
    "model_stats.csv",
]


def main() -> None:
    model_dir = "../trained_models"
    backend_dir = "../../backend/trained_models"

    os.makedirs(backend_dir, exist_ok=True)

    missing = [f for f in FILES_TO_COPY if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"ERROR: missing expected files in {model_dir}: {missing}")
        print("Run train_model.py and evaluate_model.py first.")
        return

    for filename in FILES_TO_COPY:
        shutil.copy(os.path.join(model_dir, filename), os.path.join(backend_dir, filename))

    print(f"Copied all model files to: {backend_dir}/")
    print("Backend is ready to run.")


if __name__ == "__main__":
    main()
