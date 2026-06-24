"""
CARVIS Backend - FastAPI Application
-----------------------------------------
Serves the trained XGBoost owner-vs-intruder classifier and exposes
endpoints for the React frontend to call.

Inference goes through scaler.transform() + model.predict_proba(), with
an OWNER/INTRUDER label derived from a probability threshold -- never
Isolation Forest / anomaly scores (removed entirely).

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CARVIS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")


def load_decision_threshold() -> float:
    """Use the threshold saved by evaluate_model.py so backend matches training."""
    stats_path = os.path.join(MODEL_DIR, "model_stats.csv")
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
        if "decision_threshold" in stats_df.columns and len(stats_df) > 0:
            return float(stats_df.iloc[0]["decision_threshold"])
    return 0.55


# ---- Load trained artifacts at startup ----
model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns: list[str] = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
owner_driver_id: str = joblib.load(os.path.join(MODEL_DIR, "owner_driver.pkl"))
DECISION_THRESHOLD = load_decision_threshold()

# ---- Held-out test split: real feature windows with ground truth labels ----
test_pool_path = os.path.join(MODEL_DIR, "test_split.csv")
test_pool = pd.read_csv(test_pool_path) if os.path.exists(test_pool_path) else None

# ---- In-memory session scoreboard (resets when server restarts) ----
scoreboard = {"total": 0, "correct": 0}


def score_trip(trip_df: pd.DataFrame) -> tuple[str, float]:
    """
    Scores all windows from one trip and uses the mean owner probability.
    Trip-level aggregation is far more stable than a single 10-second window.
    """
    X = trip_df[feature_columns].to_numpy(dtype=float)
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    mean_probability = float(np.mean(probabilities))
    label = "owner" if mean_probability >= DECISION_THRESHOLD else "intruder"
    return label, mean_probability


def score_window(row: pd.Series) -> tuple[str, float]:
    """Scores a single feature window (kept for compatibility)."""
    X = row[feature_columns].to_numpy(dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probability = float(model.predict_proba(X_scaled)[0, 1])
    label = "owner" if probability >= DECISION_THRESHOLD else "intruder"
    return label, probability


@app.get("/")
def root():
    return {
        "project": "CARVIS",
        "full_name": "Continuous Automotive Real-time Vehicle Intrusion System",
        "status": "online",
        "owner_driver_id": owner_driver_id,
        "model_type": "XGBClassifier",
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.get("/api/scoreboard")
def get_scoreboard():
    accuracy = (scoreboard["correct"] / scoreboard["total"] * 100) if scoreboard["total"] > 0 else 0
    return {
        "total_tests": scoreboard["total"],
        "correct_predictions": scoreboard["correct"],
        "accuracy_percent": round(accuracy, 1),
    }


@app.post("/api/scoreboard/reset")
def reset_scoreboard():
    scoreboard["total"] = 0
    scoreboard["correct"] = 0
    return {"message": "Scoreboard reset"}


@app.get("/api/test-random")
def test_random_driver():
    """
    Picks a random held-out TRIP (all windows from one unseen drive),
    scores it with mean window probability, and compares to ground truth.
    Trip-level scoring is much more reliable than a single random window.
    """
    if test_pool is None or len(test_pool) == 0:
        raise HTTPException(
            status_code=500,
            detail="Test pool not found. Run train_model.py, evaluate_model.py, "
                   "then export_test_pool.py in the model folder first.",
        )

    trip_keys = test_pool[["driver_id", "trip_folder"]].drop_duplicates()
    chosen = trip_keys.sample(n=1).iloc[0]
    trip_df = test_pool[
        (test_pool["driver_id"] == chosen["driver_id"])
        & (test_pool["trip_folder"] == chosen["trip_folder"])
    ]
    representative = trip_df.iloc[len(trip_df) // 2]

    predicted_label, probability = score_trip(trip_df)

    actual_driver = chosen["driver_id"]
    actual_behavior = representative["behavior"]
    true_label = "owner" if actual_driver == owner_driver_id else "intruder"
    was_correct = predicted_label == true_label

    scoreboard["total"] += 1
    if was_correct:
        scoreboard["correct"] += 1

    return {
        "sensor_window": {
            "accel_x_std": round(float(representative["accel_x_kf_std"]), 4),
            "accel_y_std": round(float(representative["accel_y_kf_std"]), 4),
            "yaw_std": round(float(representative["yaw_std"]), 4),
            "harsh_accel_count": int(representative["harsh_accel_count"]),
            "harsh_turn_count": int(representative["harsh_turn_count"]),
            "speed_mean": round(float(representative["speed_mean"]), 1)
            if pd.notna(representative["speed_mean"])
            else None,
            "windows_scored": int(len(trip_df)),
        },
        "prediction": {
            "label": predicted_label,
            "confidence_score": round(probability, 4),
        },
        "ground_truth": {
            "actual_driver_id": actual_driver,
            "actual_behavior": actual_behavior,
            "true_label": true_label,
        },
        "was_correct": was_correct,
        "owner_driver_id": owner_driver_id,
    }


@app.get("/api/model-stats")
def model_stats():
    """
    Returns the full evaluation metrics (precision, recall, ROC-AUC, etc.)
    pre-computed and saved by the model folder's evaluate_model.py.
    """
    stats_path = os.path.join(MODEL_DIR, "model_stats.csv")
    if not os.path.exists(stats_path):
        raise HTTPException(
            status_code=500,
            detail="model_stats.csv not found. Run evaluate_model.py then "
                   "export_test_pool.py in the model folder first.",
        )
    stats_df = pd.read_csv(stats_path)
    return stats_df.to_dict(orient="records")[0]
