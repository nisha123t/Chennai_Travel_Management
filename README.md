# Chennai Travel Management: Multimodal Route Planner

Plans journeys across Chennai Metro, Suburban Rail / MRTS, MTC buses, walking, autos and cabs.
Returns up to 3 distinct route options with time, fare, walking distance and changes, using
journey times predicted by Random Forest / Linear Regression models.

```
backend/    FastAPI + routing engine + ML models   (see backend/API.md)
frontend/   React (Vite) dashboard
```

## Run it

Easiest: open the repo in **GitHub Codespaces** (green *Code* button → *Codespaces* → *Create codespace*).
Python and Node are already installed there. Run the two parts below in two terminals
(Terminal → New Terminal, or the **+** in the terminal panel). When `npm run dev` starts,
click **Open in Browser** for port 5173. Only port 5173 needs to be opened; it forwards
`/api` calls to the backend on port 8000.

**Backend** (terminal 1)
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux/Codespaces: source .venv/bin/activate
pip install -r requirements.txt
cd models && python train_models.py && cd ..     # first time only, ~15 s
uvicorn main:app --reload --port 8000             # http://localhost:8000/docs
```

**Frontend** (terminal 2)
```bash
cd frontend
npm install
npm run dev                                       # http://localhost:5173
```

**Tests**
```bash
cd backend
python -m pytest -q
```

## Results (test set, routes unseen in training)

| Model | MAE (min) | RMSE | R² |
|---|---|---|---|
| Timetable only | 4.59 | 8.44 | 0.875 |
| Multiple Linear Regression | 5.66 | 8.06 | 0.886 |
| Random Forest | **1.81** | **3.63** | **0.977** |

Journey-time targets are simulated from timetable data (see `backend/API.md`).
