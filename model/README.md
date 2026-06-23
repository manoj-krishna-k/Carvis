# CARVIS Model Folder — Real Data Version

## WHERE TO PUT YOUR DATASET (do this first)

1. Unzip your downloaded UAH-DriveSet.
2. Copy the WHOLE unzipped folder (the one containing D1, D2, D3...) into:
   ```
   model/data/UAH-DRIVESET-v1/
   ```
   So the final path looks like:
   ```
   model/data/UAH-DRIVESET-v1/D1/20151110175712-Aggressive-MOTORWAY/RAW_ACCELEROMETERS.txt
   model/data/UAH-DRIVESET-v1/D2/...
   ```

3. If your unzipped folder has a DIFFERENT name (not "UAH-DRIVESET-v1"),
   either rename it to match, OR open `src/load_real_data.py` and change
   this line near the bottom to match your actual folder name:
   ```python
   DATASET_ROOT = "../data/UAH-DRIVESET-v1"
   ```

   NOTE: when you unzip the dataset, you may get a folder INSIDE a folder
   with the same name (e.g. `UAH-DRIVESET-v1/UAH-DRIVESET-v1/D1/...`).
   Always point DATASET_ROOT at the folder that DIRECTLY contains D1, D2, etc.

   Trip subfolder names look like:
   `20151111135612-13km-D1-DROWSY-SECONDARY`
   (format: TIMESTAMP-DISTANCEkm-DRIVERID-BEHAVIOR-ROAD)

## HOW TO RUN (in order)

```
cd model/src
pip install pandas scikit-learn joblib

python load_real_data.py       # Step 1: scans dataset, combines into one CSV
python extract_features.py     # Step 2: converts raw data into behavior features
python train_model.py          # Step 3: trains model on one driver as "the owner"
python evaluate_model.py       # Step 4: prints precision/recall/confusion matrix
```

## IF load_real_data.py GIVES ERRORS

The script prints exactly what it found vs what it expected. Common issues:

- **"Path does not exist"** -> your folder name doesn't match, fix DATASET_ROOT
- **"No driver folders (D1, D2...) found"** -> check you copied the right folder level
- **"Folder name doesn't match expected pattern"** -> your trip folder names are
  named differently than `20151110175712-Aggressive-MOTORWAY`. Paste me the
  actual folder name and I'll adjust the pattern matching.

## CHOOSING WHO IS "THE OWNER"

Open `train_model.py` and change this line to pick any driver (D1 through D6,
depending on how many drivers your download has):
```python
OWNER_DRIVER_ID = "D1"
```

## OUTPUT FILES (created automatically)

```
data/combined_raw.csv         <- all raw sensor data combined, one row per reading
data/all_features.csv         <- behavior features per time window (this is what the model uses)
trained_models/isolation_forest.pkl    <- the trained model
trained_models/feature_columns.pkl     <- list of feature names (used by backend later)
trained_models/owner_driver_id.pkl     <- which driver was treated as the owner
```

These three files in `trained_models/` are what the FastAPI backend will load
next — once this model folder is working, send me the output of
`evaluate_model.py` and we move to building the backend.
