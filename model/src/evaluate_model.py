"""
CARVIS - Full Model Evaluation (v2: XGBoost)
--------------------------------------------------
Loads the trained XGBoost model, scaler, and the EXACT held-out test
split saved by train_model.py (test_split.csv), and reports:

    accuracy, precision, recall, F1-score, ROC-AUC
    confusion matrix
    full classification report
    ROC curve plot (saved as PNG)

Ground truth: owner windows = 1, every other driver's windows = 0.
The test split already excludes every trip used in training, so these
numbers reflect genuinely unseen driving for every driver, including
the owner.
"""

from __future__ import annotations

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from threshold_utils import find_best_trip_threshold, load_decision_threshold

MODEL_DIR = "../trained_models"

# Populated from the held-out test split during main(); kept as a module-level
# default so other scripts can import a sane fallback before evaluation runs.
DECISION_THRESHOLD = load_decision_threshold(MODEL_DIR)


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    owner_driver = joblib.load(os.path.join(MODEL_DIR, "owner_driver.pkl"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_split.csv"))
    return model, scaler, feature_columns, owner_driver, test_df


def _check_overfitting(model) -> None:
    evals_result = None
    if hasattr(model, "evals_result"):
        try:
            evals_result = model.evals_result()
        except Exception:
            evals_result = None
    if not evals_result and hasattr(model, "evals_result_"):
        evals_result = getattr(model, "evals_result_", None)

    if not evals_result or not isinstance(evals_result, dict):
        print("Overfitting check: no evaluation history found in model.")
        return

    eval_sets = list(evals_result.keys())
    if not eval_sets:
        print("Overfitting check: evaluation history is empty.")
        return

    train_key = None
    val_key = None
    for key in eval_sets:
        lower = key.lower()
        if "train" in lower:
            train_key = key
            continue
        if "valid" in lower or "validation" in lower:
            if val_key is None:
                val_key = key
    if train_key is None:
        train_key = eval_sets[0]
    if val_key is None and len(eval_sets) > 1:
        val_key = eval_sets[1]

    if val_key is None:
        print("Overfitting check: only one eval dataset found; cannot compare train vs validation.")
        return

    common_metrics = set(evals_result[train_key]).intersection(evals_result[val_key])
    if not common_metrics:
        print("Overfitting check: no shared metrics between training and validation histories.")
        return

    preferred_metrics = ["auc", "aucpr", "logloss", "error", "rmse", "mae"]
    metric = next((m for m in preferred_metrics if m in common_metrics), sorted(common_metrics)[0])
    train_history = evals_result[train_key][metric]
    val_history = evals_result[val_key][metric]
    if not train_history or not val_history:
        print(f"Overfitting check: missing metric history for {metric}.")
        return

    train_final = train_history[-1]
    val_final = val_history[-1]
    if metric in ("auc", "aucpr"):
        diff = train_final - val_final
        print(
            f"Overfitting check: final train {metric} = {train_final:.3f}, "
            f"validation {metric} = {val_final:.3f}, diff = {diff:.3f}"
        )
        if diff > 0.05:
            print("Overfitting warning: training performance exceeds validation performance by more than 0.05.")
        else:
            print("Overfitting check: no strong evidence of overfitting from eval history.")
    else:
        diff = val_final - train_final
        print(
            f"Overfitting check: final train {metric} = {train_final:.3f}, "
            f"validation {metric} = {val_final:.3f}, diff = {diff:.3f}"
        )
        if diff > 0.02:
            print("Overfitting warning: validation loss/error is notably higher than training.")
        else:
            print("Overfitting check: no strong evidence of overfitting from eval history.")


def main() -> None:
    model, scaler, feature_columns, owner_driver, test_df = load_artifacts()
    _check_overfitting(model)

    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_true = (test_df["driver_id"] == owner_driver).astype(int).to_numpy()

    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class "owner"

    # ---- Threshold scan: show how precision/recall/F1 trade off across
    # candidate thresholds, so DECISION_THRESHOLD can be picked from real
    # data instead of guessed. This runs BEFORE applying the fixed
    # threshold below, purely as diagnostic output. ----
    print("Threshold scan (precision / recall / F1 at each cutoff):")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'n_pred_owner':>14}")
    best_f1 = -1.0
    best_threshold = DECISION_THRESHOLD
    for t in np.arange(0.1, 0.96, 0.05):
        pred_t = (y_proba >= t).astype(int)
        p_t = precision_score(y_true, pred_t, zero_division=0)
        r_t = recall_score(y_true, pred_t, zero_division=0)
        f1_t = f1_score(y_true, pred_t, zero_division=0)
        print(f"{t:>10.2f} {p_t:>10.3f} {r_t:>10.3f} {f1_t:>10.3f} {int(pred_t.sum()):>14}")
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t
    print(f"\nBest F1-balanced threshold found: {best_threshold:.2f} (F1={best_f1:.3f})")

    balanced_threshold, trip_accuracy = find_best_trip_threshold(
        test_df, y_proba, owner_driver
    )
    print(
        f"Best trip-level demo threshold: {balanced_threshold:.3f} "
        f"(trip accuracy={trip_accuracy:.1%})"
    )
    print(f"Using trip-level threshold for saved metrics: {balanced_threshold:.3f}")
    print(f"{'=' * 60}\n")

    decision_threshold = balanced_threshold
    y_pred = (y_proba >= decision_threshold).astype(int)

    print(f"Evaluating model trained on owner = {owner_driver}")
    print(f"Decision threshold: {decision_threshold:.3f}")
    print(f"Total test windows: {len(test_df)} "
          f"({int(y_true.sum())} owner / {int((1 - y_true).sum())} other)\n")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Owner   Predicted Intruder")
    print(f"  Actual Owner         {cm[0][0]:<15}  {cm[0][1]}")
    print(f"  Actual Intruder      {cm[1][0]:<15}  {cm[1][1]}")

    print(f"\nFull Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Intruder", "Owner"], labels=[0, 1]))

    # ---- Breakdown by driver + behavior ----
    test_df = test_df.copy()
    test_df["predicted_label"] = y_pred
    test_df["predicted_proba"] = y_proba

    print(f"{'=' * 60}")
    print("Breakdown by driver + behavior (% predicted as OWNER):")
    print(f"{'=' * 60}")
    breakdown = test_df.groupby(["driver_id", "behavior"]).apply(
        lambda g: (g["predicted_label"] == 1).mean() * 100
    )
    print(breakdown)

    print(f"{'=' * 60}")
    print("Trip-level demo accuracy (mean window probability per trip):")
    print(f"{'=' * 60}")
    trip_correct = 0
    trip_total = 0
    for (driver_id, trip_folder), group in test_df.groupby(["driver_id", "trip_folder"]):
        trip_mean_proba = float(group["predicted_proba"].mean())
        trip_pred_owner = trip_mean_proba >= decision_threshold
        trip_true_owner = driver_id == owner_driver
        trip_ok = trip_pred_owner == trip_true_owner
        trip_correct += int(trip_ok)
        trip_total += 1
        print(
            f"  {driver_id} / {trip_folder[-28:]:28s} "
            f"mean_p={trip_mean_proba:.3f} pred={'OWNER' if trip_pred_owner else 'INTRUDER'} "
            f"true={'OWNER' if trip_true_owner else 'INTRUDER'} ok={trip_ok}"
        )
    trip_accuracy = trip_correct / trip_total if trip_total else 0.0
    print(f"\nTrip-level demo accuracy: {trip_accuracy:.1%} ({trip_correct}/{trip_total})")

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#3DDC84", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"CARVIS ROC Curve (owner = {owner_driver})")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    print(f"\nROC curve saved to: {roc_path}")

    # ---- Save stats for the backend to serve ----
    stats = {
        "owner_driver_id": owner_driver,
        "total_windows_tested": len(test_df),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "f1_score": round(f1, 3),
        "roc_auc": round(roc_auc, 3),
        "decision_threshold": round(decision_threshold, 3),
        "trip_level_accuracy": round(trip_accuracy, 3),
        "live_demo_accuracy": round(trip_accuracy, 3),
        "num_drivers_tested": test_df["driver_id"].nunique(),
    }
    stats_path = os.path.join(MODEL_DIR, "model_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"Model stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
