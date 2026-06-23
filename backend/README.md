# CARVIS Backend (FastAPI)

## WHERE THIS GOES

This `backend/` folder sits next to your `model/` folder:
```
carvis_project/
├── model/
└── backend/        <- you are here
```

## SETUP ORDER (important)

The backend needs files that only exist AFTER you've trained the model.
Run things in this exact order:

1. Finish the model folder steps (load_real_data.py, extract_features.py,
   train_model.py, evaluate_model.py) until evaluate_model.py shows good results.

2. Run the NEW script in the model folder:
   ```
   cd model/src
   python export_test_pool.py
   ```
   This creates `test_pool.csv` and `model_stats.csv`, and AUTOMATICALLY
   copies everything the backend needs into `backend/trained_models/`.
   You don't need to manually copy any files yourself.

3. Now start the backend:
   ```
   cd backend/app
   pip install fastapi uvicorn
   uvicorn main:app --reload --port 8000
   ```

4. Open http://127.0.0.1:8000/docs in your browser — this is FastAPI's
   automatic interactive API documentation. You can test every endpoint
   directly here before connecting the frontend.

## ENDPOINTS

- `GET /` — health check, shows project info
- `GET /api/test-random` — picks a random unseen driving window, returns
  prediction + ground truth (this powers the "Test Random Driver" button)
- `GET /api/scoreboard` — running accuracy across the demo session
- `POST /api/scoreboard/reset` — resets the scoreboard to 0
- `GET /api/model-stats` — precision/recall/accuracy from evaluation

## IF YOU SEE "FileNotFoundError: trained_models/isolation_forest.pkl"

This means you skipped step 2 above. Go back to the model folder and run
`export_test_pool.py` — it copies the required files here automatically.
