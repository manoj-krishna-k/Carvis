"""
CARVIS - Feature Extraction (v3)
--------------------------------------
Converts raw, non-overlapping time windows of driving telemetry into
behavioral features tuned for driver identity (not driving mood).

v3 changes vs v2:
    - Uses Kalman-filtered accelerometer axes (less phone-mount noise)
    - Drops energy_* features (dominated splits without stable driver signal)
    - Adds jerk + axis-ratio features (acceleration/braking/steering rhythm)
    - Keeps a compact stat set (mean/std/rms/skew) to reduce overfitting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import skew

WINDOW_SIZE = 100  # non-overlapping window length, in raw sensor rows

# Kalman-filtered axes are cleaner for cross-driver comparison than raw accel.
SIGNALS = ["accel_x_kf", "accel_y_kf", "accel_z_kf", "yaw", "speed"]

HARSH_ACCEL_THRESHOLD_G = 0.3
HARSH_BRAKE_THRESHOLD_G = -0.3
HARSH_TURN_THRESHOLD_G = 0.3


def _safe_stat(func: Callable[[np.ndarray], float], values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    try:
        result = func(values)
        return float(result) if np.isfinite(result) else np.nan
    except Exception:
        return np.nan


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


STAT_FUNCTIONS: list[tuple[str, Callable[[np.ndarray], float]]] = [
    ("mean", np.mean),
    ("std", np.std),
    ("rms", _rms),
    ("skew", lambda v: skew(v) if len(v) > 2 else np.nan),
]


def build_feature_columns() -> list[str]:
    columns: list[str] = []
    for signal in SIGNALS:
        for suffix, _ in STAT_FUNCTIONS:
            columns.append(f"{signal}_{suffix}")

    columns.extend([
        "yaw_range",
        "speed_range",
        "harsh_accel_count",
        "harsh_brake_count",
        "harsh_turn_count",
        "jerk_x_mean",
        "jerk_y_mean",
        "jerk_z_mean",
        "lat_long_ratio",
        "brake_accel_ratio",
    ])
    return columns


FEATURE_COLUMNS: list[str] = build_feature_columns()


@dataclass
class WindowFeatures:
    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return self.values


def _compute_signal_stats(signal_name: str, series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=float)
    values = values[~np.isnan(values)]

    stats: dict[str, float] = {}
    for suffix, func in STAT_FUNCTIONS:
        stats[f"{signal_name}_{suffix}"] = _safe_stat(func, values)
    return stats


def _mean_abs_jerk(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(values))))


def _compute_engineered_features(window: pd.DataFrame) -> dict[str, float]:
    yaw = window["yaw"].to_numpy(dtype=float)
    speed = window["speed"].to_numpy(dtype=float)
    speed = speed[~np.isnan(speed)]

    accel_x = window["accel_x_kf"].to_numpy(dtype=float)
    accel_y = window["accel_y_kf"].to_numpy(dtype=float)
    accel_z = window["accel_z_kf"].to_numpy(dtype=float)

    lateral = float(np.mean(np.abs(accel_x))) if len(accel_x) else np.nan
    longitudinal = float(np.mean(np.abs(accel_y))) if len(accel_y) else np.nan
    brake_events = int(np.sum(accel_y < HARSH_BRAKE_THRESHOLD_G))
    accel_events = int(np.sum(accel_y > HARSH_ACCEL_THRESHOLD_G))

    return {
        "yaw_range": float(np.max(yaw) - np.min(yaw)) if len(yaw) else np.nan,
        "speed_range": float(np.max(speed) - np.min(speed)) if len(speed) else np.nan,
        "harsh_accel_count": accel_events,
        "harsh_brake_count": brake_events,
        "harsh_turn_count": int(np.sum(np.abs(accel_x) > HARSH_TURN_THRESHOLD_G)),
        "jerk_x_mean": _mean_abs_jerk(window["accel_x_kf"]),
        "jerk_y_mean": _mean_abs_jerk(window["accel_y_kf"]),
        "jerk_z_mean": _mean_abs_jerk(window["accel_z_kf"]),
        "lat_long_ratio": lateral / longitudinal if longitudinal and longitudinal > 1e-6 else np.nan,
        "brake_accel_ratio": brake_events / accel_events if accel_events > 0 else 0.0,
    }


def extract_features(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    required = ["accel_x_kf", "accel_y_kf", "accel_z_kf", "yaw", "speed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Trip data missing required columns: {missing}. "
            "Re-run load_real_data.py to regenerate combined_raw.csv."
        )

    num_windows = len(df) // window_size
    rows: list[dict[str, float]] = []

    for i in range(num_windows):
        window = df.iloc[i * window_size: (i + 1) * window_size]
        row: dict[str, float] = {}
        for signal in SIGNALS:
            row.update(_compute_signal_stats(signal, window[signal]))
        row.update(_compute_engineered_features(window))
        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def extract_features_per_trip(combined_df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    all_features = []

    for (_, trip), group in combined_df.groupby(["driver_id", "trip_folder"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        feats = extract_features(group, window_size=window_size)
        if len(feats) == 0:
            continue

        feats["driver_id"] = group["driver_id"].iloc[0]
        feats["behavior"] = group["behavior"].iloc[0]
        feats["road"] = group["road"].iloc[0]
        feats["trip_folder"] = trip
        all_features.append(feats)

    if not all_features:
        raise ValueError("No windows were extracted from any trip. Check window_size vs trip lengths.")

    return pd.concat(all_features, ignore_index=True)


if __name__ == "__main__":
    combined = pd.read_csv("../data/combined_raw.csv")

    print(f"Loaded combined data: {len(combined)} rows")
    print(f"Drivers: {sorted(combined['driver_id'].unique())}")
    print(f"Window size: {WINDOW_SIZE} rows")
    print(f"Total feature columns: {len(FEATURE_COLUMNS)}")

    final_features = extract_features_per_trip(combined, window_size=WINDOW_SIZE)
    final_features.to_csv("../data/all_features.csv", index=False)

    print("\nFeature extraction done!")
    print(f"Total windows extracted: {len(final_features)}")
    print("\nBreakdown by driver:")
    print(final_features["driver_id"].value_counts())
