# Chennai Travel Management: Multimodal Route Planner

Plans journeys across Chennai Metro, Suburban Rail / MRTS, MTC buses, walking, autos and cabs.
Returns up to 3 distinct route options with time, fare, walking distance and changes, using
journey times predicted by Random Forest / Linear Regression models.

```
backend/    FastAPI + routing engine + ML models   (see backend/API.md)
frontend/   React (Vite) dashboard
```

## Run it

Install Python 3.10+ and Node.js 18+ first. Run the backend and frontend in separate terminals.
The Vite development server forwards `/api` requests to FastAPI on port 8000.

**Windows PowerShell, terminal 1**
```bash
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

**Windows PowerShell, terminal 2**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` to use the planner and `http://localhost:8000/docs` for the API.
In Codespaces, use the same commands and open the forwarded frontend port (5173) in the browser.

**macOS / Linux / Codespaces backend** (terminal 1)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The saved models are included. To retrain them after changing the training data, see the
retraining command below.

**Tests** (from the repository root)
```bash
cd backend
python -m pytest -q
```

## Results (test set, routes unseen in training)

| Model | MAE (min) | RMSE | R² |
|---|---|---|---|
| Timetable only | 4.59 | 8.44 | 0.875 |
| Multiple Linear Regression | 5.66 | 8.06 | 0.886 |
| Random Forest (tuned) | **1.78** | **3.53** | **0.978** |

Random Forest settings were chosen by grid search (36 combinations, grouped cross-validation on the
training routes): cross-validated MAE 2.43 → 2.32 min. Use `python train_models.py --no-tune` to skip tuning.

Journey-time targets are simulated from timetable data (see `backend/API.md`).
