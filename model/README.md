# CARVIS Model Folder (v2 — XGBoost)

## WHAT CHANGED FROM v1

The model is now an **XGBoost binary classifier** instead of Isolation
Forest. Reasons and tradeoffs:

- v1 (Isolation Forest) trained ONLY on the owner's data -- no labeled
  intruder examples needed, "anything unfamiliar gets flagged." Honest
  pitch, but in testing it under-performed on real noisy sensor data.
- v2 (XGBoost) trains on owner (label=1) vs every other driver (label=0).
  Needs labeled "other driver" data during training, but is generally
  more accurate when that data is available -- which it is, since
  UAH-DriveSet has 6 drivers.

This is a genuine tradeoff worth knowing for your presentation: v2 is a
proper binary classifier, not a pure one-class anomaly detector.

## WHERE TO PUT YOUR DATASET (unchanged from before)

```
model/data/UAH-DRIVESET-v1/D1/<trip folder>/RAW_ACCELEROMETERS.txt
model/data/UAH-DRIVESET-v1/D1/<trip folder>/RAW_GPS.txt
model/data/UAH-DRIVESET-v1/D2/...
```

If your unzipped folder has a different name, edit `DATASET_ROOT` at the
bottom of `load_real_data.py`.

## PIPELINE FILES (run in this order)

```
cd model/src
pip install pandas numpy scipy scikit-learn xgboost joblib matplotlib

python load_real_data.py       # unchanged -- scans dataset, fixes the
                                # accel/GPS timestamp-merge bug, saves
                                # combined_raw.csv
python extract_features.py     # NEW -- 58 auto-generated statistical +
                                # engineered features per 100-row window
python train_model.py          # NEW -- trip-based split (every driver's
                                # trips split independently, no leakage),
                                # StandardScaler + XGBClassifier
python evaluate_model.py       # NEW -- accuracy/precision/recall/F1/
                                # ROC-AUC, confusion matrix, ROC curve PNG
python export_test_pool.py     # copies artifacts into backend/trained_models/
python predict.py              # optional smoke test -- scores one random
                                # held-out window
```

## KEY DESIGN DECISIONS

**Window size**: 100 raw rows, non-overlapping. At ~10Hz that's ~10
seconds per window. Incomplete trailing windows are skipped.

**Features**: for each of accel_x, accel_y, accel_z, yaw, speed --
mean, std, min, max, median, p25, p75, rms, skew, kurtosis. Plus
yaw_range, speed_range, harsh_accel_count, harsh_brake_count,
harsh_turn_count, and signal energy for the three accel axes.
`FEATURE_COLUMNS` in `extract_features.py` is generated automatically --
never hardcoded elsewhere.

**No data leakage**: `data_split.py` splits EVERY driver's trips
independently into train/test (default 70/30). No trip ever appears on
both sides. This means the owner has some trips in train and some held
out in test, and so does every other driver -- so both precision (did
we wrongly flag the owner?) and recall (did we catch the intruder?) are
measured on genuinely unseen trips.

**Decision threshold**: loaded automatically from `model_stats.csv`
(produced by `evaluate_model.py`). The backend no longer hardcodes 0.9,
which was far too strict for XGBoost probabilities on real driving data and
caused almost every window — including the owner — to be labeled INTRUDER.
Change the threshold by re-running `evaluate_model.py` and
`export_test_pool.py`, or override it temporarily in `predict.py`.

## CHOOSING WHO IS "THE OWNER"

Edit this line in `train_model.py`:
```python
OWNER_DRIVER: str = "D1"
```

## OUTPUT FILES (created automatically)

```
data/combined_raw.csv             <- all raw sensor data combined
data/all_features.csv             <- 58 features per window, all drivers/trips
trained_models/xgb_model.pkl      <- trained XGBoost classifier
trained_models/scaler.pkl         <- fitted StandardScaler
trained_models/feature_columns.pkl
trained_models/owner_driver.pkl
trained_models/test_split.csv     <- the exact held-out test set (reused by evaluate_model.py)
trained_models/model_stats.csv    <- precision/recall/F1/ROC-AUC (read by backend)
trained_models/roc_curve.png      <- ROC curve plot
```

`export_test_pool.py` copies the artifacts the backend needs into
`backend/trained_models/` automatically -- no manual copying required.
