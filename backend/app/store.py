"""Filesystem persistence: data/datasets/<id>.{csv,json} and data/runs/<id>/*."""

import json
import os
import threading
import uuid
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("PLAYGROUND_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
DATASETS = DATA_DIR / "datasets"
RUNS = DATA_DIR / "runs"

_lock = threading.Lock()  # the API and the training worker touch the same files


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def _read_json(path: Path):
    with _lock:
        return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        path.write_text(json.dumps(data), encoding="utf-8")


# --- datasets ---

def save_dataset(name: str, content: bytes, df: pd.DataFrame) -> dict:
    dataset_id = new_id()
    DATASETS.mkdir(parents=True, exist_ok=True)
    (DATASETS / f"{dataset_id}.csv").write_bytes(content)
    meta = {"id": dataset_id, "name": name, "n_rows": len(df), "columns": list(df.columns)}
    _write_json(DATASETS / f"{dataset_id}.json", meta)
    return meta


def dataset_exists(dataset_id: str) -> bool:
    return (DATASETS / f"{dataset_id}.json").exists()


def get_dataset_meta(dataset_id: str) -> dict:
    return _read_json(DATASETS / f"{dataset_id}.json")


def load_dataset(dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS / f"{dataset_id}.csv")


def list_datasets() -> list[dict]:
    return [_read_json(p) for p in sorted(DATASETS.glob("*.json"), key=os.path.getmtime, reverse=True)]


# --- runs ---

def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def run_exists(run_id: str) -> bool:
    return (run_dir(run_id) / "run.json").exists()


def get_run(run_id: str) -> dict:
    return _read_json(run_dir(run_id) / "run.json")


def save_run(run: dict) -> None:
    _write_json(run_dir(run["id"]) / "run.json", run)


def update_run(run_id: str, **fields) -> dict:
    run = get_run(run_id) | fields  # only the worker thread updates runs, so read-modify-write is safe
    save_run(run)
    return run


def list_runs() -> list[dict]:
    return sorted((_read_json(p) for p in RUNS.glob("*/run.json")), key=lambda r: r["created_at"], reverse=True)


def save_artifact(run_id: str, name: str, data) -> None:
    _write_json(run_dir(run_id) / f"{name}.json", data)


def get_artifact(run_id: str, name: str, default=None):
    path = run_dir(run_id) / f"{name}.json"
    return _read_json(path) if path.exists() else default
