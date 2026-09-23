import asyncio
import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import store
from app.ml.profile import profile_dataset
from app.runner import TERMINAL, runner
from app.schemas import RunConfig

FRONTEND_BUILD = Path(__file__).resolve().parents[2] / "frontend" / "out"


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner.start()
    yield


app = FastAPI(title="ML Playground", lifespan=lifespan)
# Only needed when running the Next.js dev server separately.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


def _require_dataset(dataset_id: str) -> None:
    if not store.dataset_exists(dataset_id):
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")


def _require_run(run_id: str) -> None:
    if not store.run_exists(run_id):
        raise HTTPException(404, f"Run '{run_id}' not found.")


# --- datasets ---

@app.post("/api/datasets")
async def upload_dataset(file: UploadFile) -> dict:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as error:
        raise HTTPException(400, f"Could not parse CSV: {error}")
    if df.empty or len(df.columns) < 2:
        raise HTTPException(400, "CSV needs at least 2 columns and 1 row.")
    return store.save_dataset(file.filename or "dataset.csv", content, df)


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    return store.list_datasets()


@app.get("/api/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str) -> dict:
    _require_dataset(dataset_id)
    return store.get_dataset_meta(dataset_id) | profile_dataset(store.load_dataset(dataset_id))


# --- runs ---

@app.post("/api/runs")
def create_run(config: RunConfig) -> dict:
    _require_dataset(config.dataset_id)
    columns = store.get_dataset_meta(config.dataset_id)["columns"]
    unknown = [c for c in [config.label, *config.features] if c not in columns]
    if unknown:
        raise HTTPException(400, f"Unknown columns: {unknown}")
    if config.label in config.features:
        raise HTTPException(400, "The label cannot also be a feature.")

    run_id = store.new_id()
    store.save_run({
        "id": run_id,
        "name": config.name or run_id,
        "created_at": time.time(),
        "status": "queued",
        "config": config.model_dump(),
    })
    runner.submit(run_id)
    return store.get_run(run_id)


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return store.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    _require_run(run_id)
    return store.get_run(run_id) | {"history": runner.history(run_id)}


@app.get("/api/runs/{run_id}/predictions")
def get_predictions(run_id: str) -> dict:
    _require_run(run_id)
    predictions = store.get_artifact(run_id, "predictions")
    if predictions is None:
        raise HTTPException(404, "Predictions are available once the run has finished.")
    return predictions


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    _require_run(run_id)
    runner.cancel(run_id)
    return {"ok": True}


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """SSE: one `data:` event per epoch (replayed from epoch 1), then an `end` event with the final status."""
    _require_run(run_id)

    async def events():
        sent = 0
        while True:
            status = store.get_run(run_id)["status"]  # read before history: terminal => history is complete
            history = runner.history(run_id)
            for event in history[sent:]:
                yield f"data: {json.dumps(event)}\n\n"
            sent = len(history)
            if status in TERMINAL:
                yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if FRONTEND_BUILD.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")
