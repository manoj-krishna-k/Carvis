"""
CARVIS Backend - FastAPI Application
-----------------------------------------
Serves the trained Isolation Forest model and exposes endpoints for
the React frontend to call.

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import random

app = FastAPI(title="CARVIS API", version="1.0.0")

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load trained model + supporting files at startup ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")

model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
owner_driver_id = joblib.load(os.path.join(MODEL_DIR, "owner_driver_id.pkl"))

# ---- Load the test pool: real feature windows with ground truth labels ----
# This file is created by export_test_pool.py from the model folder
test_pool_path = os.path.join(MODEL_DIR, "test_pool.csv")
test_pool = pd.read_csv(test_pool_path) if os.path.exists(test_pool_path) else None

# ---- In-memory session scoreboard (resets when server restarts) ----
scoreboard = {"total": 0, "correct": 0}


@app.get("/")
def root():
    return {
        "project": "CARVIS",
        "full_name": "Continuous Automotive Real-time Vehicle Intrusion System",
        "status": "online",
        "owner_driver_id": owner_driver_id,
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
    Picks a random UNSEEN window from the test pool, runs it through
    the model, and returns both the prediction AND the ground truth
    so the frontend can reveal whether the model was correct.
    """
    if test_pool is None or len(test_pool) == 0:
        raise HTTPException(
            status_code=500,
            detail="Test pool not found. Run export_test_pool.py in the model folder first."
        )

    row = test_pool.sample(n=1).iloc[0]

    features_df = pd.DataFrame([row[feature_columns].to_dict()])
    prediction = model.predict(features_df)[0]          # 1 = normal/owner, -1 = anomaly/intruder
    score = model.decision_function(features_df)[0]      # confidence score

    predicted_label = "owner" if prediction == 1 else "intruder"
    actual_driver = row["driver_id"]
    actual_behavior = row["behavior"]
    true_label = "owner" if actual_driver == owner_driver_id else "intruder"

    was_correct = predicted_label == true_label

    # Update scoreboard
    scoreboard["total"] += 1
    if was_correct:
        scoreboard["correct"] += 1

    return {
        "sensor_window": {
            "accel_x_std": round(float(row["accel_x_std"]), 4),
            "accel_y_std": round(float(row["accel_y_std"]), 4),
            "yaw_std": round(float(row["yaw_std"]), 4),
            "harsh_accel_count": int(row["harsh_accel_count"]),
            "harsh_turn_count": int(row["harsh_turn_count"]),
            "speed_mean": round(float(row["speed_mean"]), 1) if pd.notna(row["speed_mean"]) else None,
        },
        "prediction": {
            "label": predicted_label,
            "confidence_score": round(float(score), 4),
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
    Returns the full evaluation metrics (precision, recall, etc.)
    pre-computed and saved by the model folder's evaluate_model.py.
    """
    stats_path = os.path.join(MODEL_DIR, "model_stats.csv")
    if not os.path.exists(stats_path):
        raise HTTPException(
            status_code=500,
            detail="model_stats.csv not found. Run export_test_pool.py in the model folder first."
        )
    stats_df = pd.read_csv(stats_path)
    return stats_df.to_dict(orient="records")[0]
