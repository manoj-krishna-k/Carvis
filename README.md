# CARVIS
### Continuous Automotive Real-time Vehicle Intrusion System

A behavioral-fingerprinting vehicle security system. Instead of tracking
*where* a car is (GPS), CARVIS learns *how* the owner drives — acceleration
rhythm, braking sharpness, turning style — and flags the moment someone
else's driving pattern shows up.

Built on real driving telemetry (UAH-DriveSet), an Isolation Forest
anomaly detection model, a FastAPI backend, and a React dashboard.

---

## Project Structure

```
carvis_project/
├── model/      <- Data loading, feature engineering, model training & evaluation
├── backend/    <- FastAPI server that serves the trained model
└── frontend/   <- React dashboard for the live demo
```

Each folder has its own README with detailed steps. Follow them **in this
exact order** — each stage produces files the next stage needs.

---

## Full Setup Order (start to finish)

### 1. Model folder
```
cd model/src
pip install pandas scikit-learn joblib

python load_real_data.py
python extract_features.py
python train_model.py
python evaluate_model.py
python export_test_pool.py   <- this copies everything the backend needs automatically
```
See `model/README.md` for where exactly to place your downloaded dataset.

### 2. Backend
```
cd backend/app
pip install fastapi uvicorn
uvicorn main:app --reload --port 8000
```
Leave this terminal open. Visit `http://127.0.0.1:8000/docs` to test the
API directly before connecting the frontend.

### 3. Frontend
```
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173`).

---

## Demo Flow for Judges

1. Open the dashboard — explain the concept in one line: *"the car learns
   how the owner drives, not where the car is."*
2. Click **"Test Random Driver"** a few times. Each click pulls a real,
   unseen window from the dataset — neither you nor the model know in
   advance whether it's the owner or another driver.
3. Point at the scoreboard — it's accumulating live, in front of them.
4. Scroll to **Model Performance** — show precision/recall as proof,
   not just a claim.
5. Use the **"How this works"** section to walk through the pipeline:
   real data → feature engineering → Isolation Forest → live verdict.

---

## If Something Breaks Mid-Demo

- **Frontend shows "Could not reach the CARVIS backend"** — the FastAPI
  terminal probably stopped. Restart it with the Step 2 command above.
- **Backend won't start, missing .pkl files** — you skipped
  `export_test_pool.py` in the model folder. Run it, it copies the
  required files into `backend/trained_models/` automatically.
