"""
CARVIS - Diagnose Why Owner vs Intruder Separation Is Weak
------------------------------------------------------------------
Run this AFTER extract_features.py (needs all_features.csv) to find out
WHY the owner-vs-intruder model has low F1/AUC, instead of guessing.

Checks four things:
    1. Does driver_id or behavior explain more feature variance?
       (if behavior dominates, the model may be partly learning
       "aggressive vs normal" rather than "D1 vs everyone else")
    2. Top XGBoost feature importances -- which features is the model
       actually relying on?
    3. Per-driver mean/std of the most-important features -- does D1
       genuinely look different from D2-D6, or do they overlap heavily?
    4. Outlier check on energy_* features, which can have very different
       scale than std-based features and may dominate splits.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from extract_features import FEATURE_COLUMNS

MODEL_DIR = "../trained_models"


def variance_explained_by_group(df: pd.DataFrame, group_col: str, feature_cols: list[str]) -> float:
    """
    Rough eta-squared (variance explained) for how much `group_col`
    explains variation in `feature_cols`, averaged across all features.
    Higher = that grouping explains more of the feature variation.
    """
    scores = []
    for col in feature_cols:
        values = df[col].to_numpy(dtype=float)
        valid = ~np.isnan(values)
        if valid.sum() < 4:
            continue
        groups = df.loc[valid, group_col].to_numpy()
        vals = values[valid]

        overall_mean = vals.mean()
        ss_total = np.sum((vals - overall_mean) ** 2)
        if ss_total == 0:
            continue

        temp = pd.DataFrame({"v": vals, "g": groups})
        group_means = temp.groupby("g")["v"].mean()
        group_counts = temp.groupby("g")["v"].count()
        ss_between = float(np.sum(group_counts * (group_means - overall_mean) ** 2))

        eta_sq = ss_between / ss_total
        scores.append(eta_sq)

    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    df = pd.read_csv("../data/all_features.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    print(f"{'=' * 70}")
    print("CHECK 1: Does driver_id or behavior explain more feature variance?")
    print(f"{'=' * 70}")
    eta_driver = variance_explained_by_group(df, "driver_id", FEATURE_COLUMNS)
    eta_behavior = variance_explained_by_group(df, "behavior", FEATURE_COLUMNS)
    print(f"  Avg variance explained by driver_id : {eta_driver:.4f}")
    print(f"  Avg variance explained by behavior   : {eta_behavior:.4f}")
    if eta_behavior > eta_driver:
        print("  -> WARNING: behavior explains MORE variance than driver identity.")
        print("     This means windows cluster by driving STYLE-OF-THE-MOMENT")
        print("     (aggressive/normal/drowsy) more than by WHO is driving.")
        print("     This directly limits how well any model can separate drivers.")
    else:
        print("  -> driver_id explains more variance than behavior, which is good:")
        print("     it means there IS a learnable per-driver signature in this feature set.")

    print(f"\n{'=' * 70}")
    print("CHECK 2: Top 15 XGBoost feature importances")
    print(f"{'=' * 70}")
    try:
        model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        top15 = importances.sort_values(ascending=False).head(15)
        print(top15.to_string())
    except FileNotFoundError:
        print("  xgb_model.pkl not found -- run train_model.py first.")
        top15 = None

    if top15 is not None:
        print(f"\n{'=' * 70}")
        print("CHECK 3: Per-driver mean of top 5 most important features")
        print(f"{'=' * 70}")
        top5_features = top15.head(5).index.tolist()
        owner_driver = joblib.load(f"{MODEL_DIR}/owner_driver.pkl")
        summary = df.groupby("driver_id")[top5_features].mean()
        print(summary.round(4))
        print(f"\n  (owner driver = {owner_driver} -- compare its row to the others above.")
        print("   If the owner's row sits inside the range of other drivers' rows")
        print("   for most of these features, that's why separation is weak.)")

    print(f"\n{'=' * 70}")
    print("CHECK 4: Scale comparison -- energy_* features vs std-based features")
    print(f"{'=' * 70}")
    energy_cols = [c for c in FEATURE_COLUMNS if c.startswith("energy_")]
    std_cols = [c for c in FEATURE_COLUMNS if c.endswith("_std")]
    for col in energy_cols + std_cols:
        print(f"  {col:30s} mean={df[col].mean():>14.3f}  std={df[col].std():>14.3f}  "
              f"max={df[col].max():>14.3f}")
    print("\n  (If energy_* values are orders of magnitude larger than std-based")
    print("   features even after scaling, they may be dominating tree splits")
    print("   without adding real signal. Consider dropping energy_* and")
    print("   re-running train_model.py to see if F1 improves.)")


if __name__ == "__main__":
    main()
