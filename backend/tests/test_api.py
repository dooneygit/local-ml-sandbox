import json

import numpy as np
import pandas as pd
import pytest


def make_csv() -> bytes:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0, 10, n)
    color = rng.choice(["red", "blue"], n)
    df = pd.DataFrame({
        "x": x,
        "color": color,
        "noise": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),  # has missing values
        "y": 3 * x + np.where(color == "red", 5, 0) + rng.normal(0, 0.1, n),
        "big": (x > 5).astype(int),
    })
    return df.to_csv(index=False).encode()


def stream(client, run_id: str) -> tuple[list[dict], str]:
    epochs, status = [], None
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        event = "message"
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
                if event == "end":
                    status = data["status"]
                else:
                    epochs.append(data)
                event = "message"
    return epochs, status


@pytest.fixture(scope="module")
def dataset(client):
    response = client.post("/api/datasets", files={"file": ("toy.csv", make_csv(), "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_rejects_non_csv(client):
    response = client.post("/api/datasets", files={"file": ("bad.csv", b"just one column\n1\n", "text/csv")})
    assert response.status_code == 400


def test_profile(client, dataset):
    assert dataset["n_rows"] == 400
    profile = client.get(f"/api/datasets/{dataset['id']}/profile").json()
    kinds = {c["name"]: c["kind"] for c in profile["columns"]}
    assert kinds == {"x": "numeric", "color": "categorical", "noise": "numeric", "y": "numeric", "big": "numeric"}
    assert profile["correlation"]["columns"] == ["x", "noise", "y", "big"]
    assert profile["correlation"]["values"][0][0] == pytest.approx(1.0)
    assert len(profile["scatter_sample"]["x"]) == 400
    assert None in profile["scatter_sample"]["noise"]  # NaN serialized as null


def test_regression_run_streams_and_learns(client, dataset):
    config = {
        "dataset_id": dataset["id"], "label": "y", "features": ["x", "color", "noise"],
        "optimizer": "adam", "learning_rate": 0.1, "epochs": 30, "batch_size": 32,
    }
    run = client.post("/api/runs", json=config).json()
    assert run["status"] == "queued"

    epochs, status = stream(client, run["id"])
    assert status == "completed"
    assert [e["epoch"] for e in epochs] == list(range(1, 31))
    assert {"loss", "val_loss", "rmse", "val_rmse"} <= epochs[0].keys()
    assert epochs[-1]["loss"] < epochs[0]["loss"]

    result = client.get(f"/api/runs/{run['id']}").json()
    assert result["task"] == "regression"
    assert result["metrics"]["r2"] > 0.95
    assert len(result["history"]) == 30

    predictions = client.get(f"/api/runs/{run['id']}/predictions").json()
    assert len(predictions["observed"]) == len(predictions["predicted"]) == 80  # 20% validation split
    assert set(predictions["features"]) == {"x", "color", "noise"}


def test_classification_is_inferred(client, dataset):
    config = {"dataset_id": dataset["id"], "label": "big", "features": ["x"], "hidden_layers": [8],
              "optimizer": "adam", "learning_rate": 0.05, "epochs": 15}
    run = client.post("/api/runs", json=config).json()
    _, status = stream(client, run["id"])
    assert status == "completed"

    result = client.get(f"/api/runs/{run['id']}").json()
    assert result["task"] == "classification"
    assert result["classes"] == ["0", "1"]
    assert result["metrics"]["accuracy"] > 0.8
    predictions = client.get(f"/api/runs/{run['id']}/predictions").json()
    assert set(predictions["predicted"]) <= {"0", "1"}


def test_regression_on_categorical_label_fails_cleanly(client, dataset):
    config = {"dataset_id": dataset["id"], "label": "color", "features": ["x"], "task": "regression", "epochs": 1}
    run = client.post("/api/runs", json=config).json()
    _, status = stream(client, run["id"])
    assert status == "failed"
    assert "numeric label" in client.get(f"/api/runs/{run['id']}").json()["error"]


def test_cancel(client, dataset):
    config = {"dataset_id": dataset["id"], "label": "y", "features": ["x"], "epochs": 1000, "batch_size": 1}
    run = client.post("/api/runs", json=config).json()
    client.post(f"/api/runs/{run['id']}/cancel")
    epochs, status = stream(client, run["id"])
    assert status == "cancelled"
    assert len(epochs) < 1000


@pytest.mark.parametrize("overrides, code", [
    ({"features": ["y"]}, 400),             # label also a feature
    ({"features": ["missing"]}, 400),       # unknown column
    ({"features": []}, 422),
    ({"hidden_layers": [0]}, 422),
    ({"learning_rate": 0}, 422),
    ({"dataset_id": "nope"}, 404),
])
def test_run_validation(client, dataset, overrides, code):
    config = {"dataset_id": dataset["id"], "label": "y", "features": ["x"]} | overrides
    assert client.post("/api/runs", json=config).status_code == code


def test_list_runs(client):
    runs = client.get("/api/runs").json()
    assert len(runs) >= 4
    assert runs == sorted(runs, key=lambda r: r["created_at"], reverse=True)
