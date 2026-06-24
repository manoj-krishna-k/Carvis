"""
CARVIS - Prediction Interface
------------------------------------
Loads the trained XGBoost model, scaler, and feature column list, and
exposes a single function for scoring a feature window as OWNER or
INTRUDER. This is the module the FastAPI backend imports directly --
it deliberately has no training/evaluation logic, only inference.

Decision rule:
    probability = model.predict_proba(scaled_features)[:, 1]
    OWNER     if probability > DECISION_THRESHOLD
    INTRUDER  otherwise
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from threshold_utils import load_decision_threshold

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
DEFAULT_DECISION_THRESHOLD = load_decision_threshold(MODEL_DIR)


@dataclass
class PredictionResult:
    """Structured result of a single prediction."""

    probability: float
    prediction: str  # "OWNER" or "INTRUDER"

    def as_dict(self) -> dict[str, float | str]:
        return {"probability": self.probability, "prediction": self.prediction}


class CarvisPredictor:
    """
    Wraps the trained model + scaler + feature column list so callers
    (e.g. the FastAPI backend) don't need to know any loading details.
    """

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        threshold: float | None = None,
    ) -> None:
        self.model = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.feature_columns: list[str] = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
        self.owner_driver: str = joblib.load(os.path.join(model_dir, "owner_driver.pkl"))
        self.threshold = (
            threshold if threshold is not None else load_decision_threshold(model_dir)
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """
        Args:
            features: a DataFrame with exactly one row containing (at
                least) all columns in self.feature_columns.

        Returns:
            PredictionResult with the owner-class probability and the
            OWNER/INTRUDER label derived from self.threshold.
        """
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Input features missing required columns: {missing}")

        X = features[self.feature_columns].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X)

        probability = float(self.model.predict_proba(X_scaled)[0, 1])
        label = "OWNER" if probability >= self.threshold else "INTRUDER"

        return PredictionResult(probability=probability, prediction=label)

    def predict_batch(self, features: pd.DataFrame) -> list[PredictionResult]:
        """Same as predict(), but for multiple rows at once."""
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Input features missing required columns: {missing}")

        X = features[self.feature_columns].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return [
            PredictionResult(
                probability=float(p),
                prediction="OWNER" if p >= self.threshold else "INTRUDER",
            )
            for p in probabilities
        ]


if __name__ == "__main__":
    # Quick smoke test using one random row from the saved test split,
    # if it exists -- not a substitute for evaluate_model.py's full report.
    test_split_path = os.path.join(MODEL_DIR, "test_split.csv")
    if not os.path.exists(test_split_path):
        print("No test_split.csv found. Run train_model.py first.")
    else:
        predictor = CarvisPredictor()
        sample_df = pd.read_csv(test_split_path).sample(n=1, random_state=None)

        result = predictor.predict(sample_df)
        actual_driver = sample_df["driver_id"].iloc[0]
        actual_behavior = sample_df["behavior"].iloc[0]

        print(f"Sampled window from driver {actual_driver} ({actual_behavior})")
        print(f"Prediction: {result.as_dict()}")
        print(f"Ground truth: {'OWNER' if actual_driver == predictor.owner_driver else 'INTRUDER'}")
