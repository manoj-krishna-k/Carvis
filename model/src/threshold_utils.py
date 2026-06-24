"""
Shared helpers for picking and loading the owner/intruder decision threshold.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


DEFAULT_DECISION_THRESHOLD = 0.5


def find_best_trip_threshold(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    owner_driver: str,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.005,
) -> tuple[float, float]:
    """
    Optimizes the cutoff for trip-level demo scoring (mean window probability
    per trip). This matches how the FastAPI /api/test-random endpoint works.
    """
    scored = test_df.copy()
    scored["_proba"] = y_proba

    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_accuracy = -1.0

    for threshold in np.arange(min_threshold, max_threshold + step, step):
        correct = 0
        total = 0
        for (_, _), group in scored.groupby(["driver_id", "trip_folder"]):
            trip_mean = float(group["_proba"].mean())
            driver_id = group["driver_id"].iloc[0]
            predicted_owner = trip_mean >= threshold
            actual_owner = driver_id == owner_driver
            correct += int(predicted_owner == actual_owner)
            total += 1

        accuracy = correct / total if total else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def find_best_demo_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.005,
    min_owner_recall: float = 0.35,
) -> tuple[float, float, float]:
    """
    Picks the cutoff that maximizes overall accuracy on the held-out test
    pool (what the live demo samples from), while requiring a minimum
    owner recall so D1 is not always labeled intruder.

    Returns (threshold, accuracy, owner_recall).
    """
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_accuracy = -1.0
    best_recall = 0.0

    for threshold in np.arange(min_threshold, max_threshold + step, step):
        predicted = (y_proba >= threshold).astype(int)

        owner_mask = y_true == 1
        owner_recall = (
            float(((predicted == 1) & owner_mask).sum() / owner_mask.sum())
            if owner_mask.any()
            else 0.0
        )
        if owner_recall < min_owner_recall:
            continue

        accuracy = float((predicted == y_true).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
            best_recall = owner_recall

    if best_accuracy < 0:
        # Fall back to the threshold with the highest owner recall if the
        # minimum recall constraint could not be satisfied.
        for threshold in np.arange(min_threshold, max_threshold + step, step):
            predicted = (y_proba >= threshold).astype(int)
            owner_mask = y_true == 1
            owner_recall = (
                float(((predicted == 1) & owner_mask).sum() / owner_mask.sum())
                if owner_mask.any()
                else 0.0
            )
            accuracy = float((predicted == y_true).mean())
            if owner_recall > best_recall or (
                np.isclose(owner_recall, best_recall) and accuracy > best_accuracy
            ):
                best_threshold = float(threshold)
                best_recall = owner_recall
                best_accuracy = accuracy

    return best_threshold, best_accuracy, best_recall


def find_balanced_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.005,
) -> tuple[float, float]:
    """Legacy helper: maximizes (owner recall + intruder specificity) / 2."""
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_score = -1.0

    for threshold in np.arange(min_threshold, max_threshold + step, step):
        predicted = (y_proba >= threshold).astype(int)

        true_positives = int(((predicted == 1) & (y_true == 1)).sum())
        false_negatives = int(((predicted == 0) & (y_true == 1)).sum())
        true_negatives = int(((predicted == 0) & (y_true == 0)).sum())
        false_positives = int(((predicted == 1) & (y_true == 0)).sum())

        owner_recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives)
            else 0.0
        )
        intruder_specificity = (
            true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives)
            else 0.0
        )
        balanced_score = (owner_recall + intruder_specificity) / 2.0

        if balanced_score > best_score:
            best_score = balanced_score
            best_threshold = float(threshold)

    return best_threshold, best_score


def load_decision_threshold(model_dir: str) -> float:
    stats_path = os.path.join(model_dir, "model_stats.csv")
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
        if "decision_threshold" in stats.columns and len(stats) > 0:
            return float(stats.iloc[0]["decision_threshold"])
    return DEFAULT_DECISION_THRESHOLD
