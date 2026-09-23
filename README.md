# ML Playground

Upload a CSV, pick a label and features, set hyperparameters, and train a Keras model locally.
Watch loss curves live, compare runs side by side, and inspect metrics, the correlation matrix,
the scatter matrix, and predictions.

- `backend/` — FastAPI + Keras (TensorFlow). Runs train one at a time in a background worker;
  epochs stream to the browser over SSE. Everything is stored as files under `data/`.
- `frontend/` — Next.js static export + Plotly, served by FastAPI.
- `samples/taxi_train.csv` — sample dataset (try label `FARE`, features `TRIP_MILES`, `TRIP_SECONDS`).

Requires Python 3.12 and Node 20+.

## Run

```sh
python -m venv .venv
.venv/Scripts/pip install -r backend/requirements.txt   # macOS/Linux: .venv/bin/pip
cd frontend && npm install && npm run build && cd ..
cd backend && ../.venv/Scripts/python -m uvicorn app.main:app
```

Open http://localhost:8000.

## Develop

```sh
cd backend && ../.venv/Scripts/python -m uvicorn app.main:app --reload   # API on :8000
cd frontend && npm run dev                                               # UI on :3000, calls :8000
cd backend && ../.venv/Scripts/python -m pytest                          # backend tests
```
