"""
CARVIS - Feature Extraction (Real Data Version)
----------------------------------------------------
Same concept as before: break driving into small time WINDOWS and
calculate behavior patterns inside each window.

Updated to match REAL UAH-DriveSet columns:
    accel_x, accel_y, accel_z   -> raw acceleration (Gs)
    yaw                          -> replaces our old "gyro_z" (turning sharpness)
    speed                        -> real speed from GPS (Km/h)

Accelerometer data is recorded at ~10Hz (10 rows per second), so a
window_size of 50 rows = ~5 seconds of driving. Adjust if your data's
actual sample rate differs.
"""

import pandas as pd
import numpy as np


def extract_features(df, window_size=50):
    """
    Splits driver data into windows and calculates behavior features
    for each window. Expects a DataFrame with at least:
    accel_x, accel_y, accel_z, yaw, speed
    """
    features_list = []
    num_windows = len(df) // window_size

    for i in range(num_windows):
        window = df.iloc[i * window_size: (i + 1) * window_size]

        features = {
            # Smoothness of forward/backward and side-to-side acceleration
            "accel_x_mean": window["accel_x"].mean(),
            "accel_x_std": window["accel_x"].std(),
            "accel_y_mean": window["accel_y"].mean(),
            "accel_y_std": window["accel_y"].std(),
            "accel_z_std": window["accel_z"].std(),

            # Turning sharpness (yaw = direction the car is pointing)
            "yaw_std": window["yaw"].std(),
            "yaw_range": window["yaw"].max() - window["yaw"].min(),

            # Harsh event counts -- threshold based on Gs (1 G = normal gravity)
            # Anything beyond 0.3G sideways/forward is a noticeably hard event
            "harsh_accel_count": (window["accel_y"].abs() > 0.3).sum(),
            "harsh_turn_count": (window["accel_x"].abs() > 0.3).sum(),

            # Speed consistency (if GPS speed data is available)
            "speed_mean": window["speed"].mean() if window["speed"].notna().any() else np.nan,
            "speed_std": window["speed"].std() if window["speed"].notna().any() else np.nan,
        }
        features_list.append(features)

    return pd.DataFrame(features_list)


if __name__ == "__main__":
    combined = pd.read_csv("../data/combined_raw.csv")

    print(f"Loaded combined data: {len(combined)} rows")
    print(f"Drivers: {sorted(combined['driver_id'].unique())}")

    all_features = []

    # Extract features SEPARATELY per trip, so windows don't blend
    # data from two different trips together
    for (driver, trip), group in combined.groupby(["driver_id", "trip_folder"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        feats = extract_features(group, window_size=50)

        if len(feats) == 0:
            continue

        feats["driver_id"] = driver
        feats["behavior"] = group["behavior"].iloc[0]
        feats["road"] = group["road"].iloc[0]
        feats["trip_folder"] = trip

        all_features.append(feats)

    final_features = pd.concat(all_features, ignore_index=True)
    final_features.to_csv("../data/all_features.csv", index=False)

    print(f"\nFeature extraction done!")
    print(f"Total windows extracted: {len(final_features)}")
    print(f"\nBreakdown by behavior:")
    print(final_features["behavior"].value_counts())
    print(f"\nBreakdown by driver:")
    print(final_features["driver_id"].value_counts())
