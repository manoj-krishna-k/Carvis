"""
CARVIS - Real Data Loader
-----------------------------
Scans the UAH-DriveSet folder structure and loads RAW_ACCELEROMETERS.txt
(and optionally RAW_GPS.txt) from every driver/trip folder.

REAL FORMAT (confirmed from official UAH-DriveSet documentation):

RAW_ACCELEROMETERS.txt columns (space-separated, no header row):
    0: Timestamp (seconds since trip start)
    1: Activation boolean (1 if speed > 50 km/h)
    2: X acceleration (Gs)
    3: Y acceleration (Gs)
    4: Z acceleration (Gs)
    5: X acceleration filtered by Kalman Filter (Gs)
    6: Y acceleration filtered by Kalman Filter (Gs)
    7: Z acceleration filtered by Kalman Filter (Gs)
    8: Roll (degrees)
    9: Pitch (degrees)
    10: Yaw (degrees)

RAW_GPS.txt columns (space-separated, no header row):
    0: Timestamp (seconds since trip start)
    1: Speed (Km/h)
    2: Latitude
    3: Longitude
    4: Altitude
    5: Vertical accuracy
    6: Horizontal accuracy
    7: Course
    8: Difcourse (course variation)

Folder structure expected:
    UAH-DRIVESET-v1/
        D1/
            20151110175712-Aggressive-MOTORWAY/
                RAW_ACCELEROMETERS.txt
                RAW_GPS.txt
            20151111123622-Normal-MOTORWAY/
                ...
        D2/
            ...

If your folder names differ slightly, this script prints exactly what it
found so you can fix the path or naming pattern.
"""

import os
import re
import pandas as pd
import numpy as np

ACCEL_COLUMNS = [
    "timestamp", "activation_bool",
    "accel_x", "accel_y", "accel_z",
    "accel_x_kf", "accel_y_kf", "accel_z_kf",
    "roll", "pitch", "yaw"
]

GPS_COLUMNS = [
    "timestamp", "speed", "latitude", "longitude",
    "altitude", "vertical_accuracy", "horizontal_accuracy",
    "course", "difcourse"
]

# Matches folder names like: 20151111135612-13km-D1-DROWSY-SECONDARY
# Pattern: TIMESTAMP-DISTANCEkm-DRIVERID-BEHAVIOR-ROAD
TRIP_FOLDER_PATTERN = re.compile(
    r"^\d+-[\d.]+km-D\d+-(NORMAL|AGGRESSIVE|DROWSY)-(MOTORWAY|SECONDARY)$",
    re.IGNORECASE
)


def load_trip(trip_folder_path):
    """
    Loads RAW_ACCELEROMETERS.txt (required) and RAW_GPS.txt (optional)
    from a single trip folder, and merges speed into the accelerometer
    data using position-based interpolation (since GPS is 1Hz and
    accelerometer is 10Hz, and their timestamp columns don't reliably
    share a common zero point).
    """
    accel_path = os.path.join(trip_folder_path, "RAW_ACCELEROMETERS.txt")
    gps_path = os.path.join(trip_folder_path, "RAW_GPS.txt")

    if not os.path.exists(accel_path):
        print(f"  [SKIP] No RAW_ACCELEROMETERS.txt in: {trip_folder_path}")
        return None

    accel_df = pd.read_csv(accel_path, sep=r"\s+", header=None, names=ACCEL_COLUMNS)
    accel_df["timestamp"] = accel_df["timestamp"].astype(float)

    if os.path.exists(gps_path):
        gps_df = pd.read_csv(gps_path, sep=r"\s+", header=None, names=GPS_COLUMNS)
        gps_df["timestamp"] = gps_df["timestamp"].astype(float)

        # NOTE: We do NOT merge by matching timestamp VALUES across the two
        # files. In practice, the accelerometer and GPS files' timestamp
        # columns do not share a reliable common zero point, which makes
        # pd.merge_asof on raw timestamps snap every row to a single nearest
        # GPS point (std becomes 0, speed becomes a frozen, often wrong value).
        #
        # Instead, we use POSITION-based interpolation: accelerometer runs at
        # ~10Hz and GPS at ~1Hz, so we map each accelerometer row to its
        # proportional position in the GPS file's row sequence. This is robust
        # even if the two files' timestamp columns are inconsistent.
        n_accel = len(accel_df)
        n_gps = len(gps_df)

        if n_gps > 1:
            gps_positions = np.linspace(0, n_gps - 1, num=n_accel)
            speed_values = np.interp(gps_positions, np.arange(n_gps), gps_df["speed"].values)
        else:
            speed_values = np.full(n_accel, gps_df["speed"].iloc[0] if n_gps == 1 else np.nan)

        accel_df["speed"] = speed_values
    else:
        accel_df["speed"] = None  # no GPS file found, leave speed empty

    return accel_df


def scan_dataset(root_path):
    """
    Walks through D1, D2, D3... folders, finds all valid trip subfolders,
    loads their data, and tags each row with driver_id and behavior label.

    Returns one combined DataFrame with everything.
    """
    all_trips = []

    if not os.path.exists(root_path):
        print(f"ERROR: Path does not exist: {root_path}")
        print("Update DATASET_ROOT at the bottom of this file to your actual dataset folder.")
        return None

    driver_folders = sorted([
        f for f in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, f)) and f.upper().startswith("D")
    ])

    if not driver_folders:
        print(f"ERROR: No driver folders (D1, D2...) found inside {root_path}")
        print("Found these folders instead:", os.listdir(root_path))
        return None

    print(f"Found {len(driver_folders)} driver folders: {driver_folders}\n")

    for driver_folder in driver_folders:
        driver_path = os.path.join(root_path, driver_folder)
        trip_folders = [
            f for f in os.listdir(driver_path)
            if os.path.isdir(os.path.join(driver_path, f))
        ]

        for trip_folder in trip_folders:
            match = TRIP_FOLDER_PATTERN.match(trip_folder)

            if not match:
                print(f"  [WARNING] Folder name doesn't match expected pattern, skipping: {trip_folder}")
                print(f"            Expected something like: 20151110175712-Aggressive-MOTORWAY")
                continue

            behavior = match.group(1).capitalize()  # Normal / Aggressive / Drowsy
            road = match.group(2).upper()            # MOTORWAY / SECONDARY

            trip_path = os.path.join(driver_path, trip_folder)
            trip_df = load_trip(trip_path)

            if trip_df is None:
                continue

            trip_df["driver_id"] = driver_folder
            trip_df["behavior"] = behavior
            trip_df["road"] = road
            trip_df["trip_folder"] = trip_folder

            all_trips.append(trip_df)
            print(f"  [OK] Loaded {driver_folder}/{trip_folder} -> {len(trip_df)} rows ({behavior}, {road})")

    if not all_trips:
        print("\nNo valid trips were loaded. Check folder naming above.")
        return None

    combined = pd.concat(all_trips, ignore_index=True)
    return combined


if __name__ == "__main__":
    # CHANGE THIS to the actual path where you unzipped the dataset
    DATASET_ROOT = "../data/UAH-DRIVESET-v1"

    print(f"Scanning dataset at: {DATASET_ROOT}\n")
    combined_df = scan_dataset(DATASET_ROOT)

    if combined_df is not None:
        output_path = "../data/combined_raw.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"\n{'=' * 50}")
        print(f"DONE. Combined dataset saved to: {output_path}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Drivers found: {sorted(combined_df['driver_id'].unique())}")
        print(f"Behaviors found: {sorted(combined_df['behavior'].unique())}")
        print(f"{'=' * 50}")