# CARVIS Frontend (React + Vite)

## WHERE THIS GOES

This `frontend/` folder sits next to `model/` and `backend/`:
```
carvis_project/
├── model/
├── backend/
└── frontend/        <- you are here
```

## SETUP ORDER (important)

The frontend talks to the FastAPI backend on `http://127.0.0.1:8000`.
The backend must already be running before you open the frontend, otherwise
you'll see a "Could not reach the CARVIS backend" message on screen
(this is intentional -- it tells you exactly what's wrong instead of
silently failing).

### Step 1 -- Start the backend first (in one terminal)
```
cd backend/app
uvicorn main:app --reload --port 8000
```
Leave this terminal running.

### Step 2 -- Start the frontend (in a SECOND terminal)
```
cd frontend
npm install
npm run dev
```

### Step 3 -- Open the browser
Vite will print a URL, usually:
```
http://localhost:5173
```
Open that in your browser. You should see the CARVIS dashboard.

## WHAT YOU'LL SEE

- A hero section with a live animated waveform (idle/grey until you test)
- **"Test Random Driver"** button -- picks a random unseen driving window
  from the dataset, sends it to the backend, and reveals:
  - The model's prediction (Owner / Intruder)
  - The actual ground truth (which driver it really was)
  - Whether the model got it right
- A running scoreboard -- accuracy across however many times you click test
- A Model Performance panel -- real precision/recall/accuracy numbers
- A "How this works" section -- good for explaining the pipeline to judges

## FOR YOUR PRESENTATION

Click "Test Random Driver" multiple times in front of judges. Since the
window is picked randomly from real data each time, this is a genuine
live test, not a scripted demo. The scoreboard updates honestly -- if the
model gets one wrong, it shows that too.

## IF THE BACKEND CHANGES PORT

If you ever run FastAPI on a different port, update this line in
`src/api/carvis.js`:
```js
const API_BASE = "http://127.0.0.1:8000";
```

## BUILDING FOR PRODUCTION (optional, not needed for demo)

```
npm run build
```
This creates a `dist/` folder you could deploy anywhere, but for the
hackathon demo, `npm run dev` is all you need.
