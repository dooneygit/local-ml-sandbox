from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from app.ml.model import create_model
from app.ml.profile import is_numeric, jsonable
from app.schemas import RunConfig, Task

PREDICTION_ROWS = 500
MAX_INFERRED_CLASSES = 10
EVAL_BATCH_SIZE = 1024  # validation/predict batch size; independent of the training batch size


def infer_task(label: pd.Series) -> Task:
    if not is_numeric(label):
        return "classification"
    values = label.dropna()
    if values.nunique() <= MAX_INFERRED_CLASSES and (values % 1 == 0).all():
        return "classification"
    return "regression"


def _fill_values(df: pd.DataFrame, features: list[str]) -> dict:
    """Missing-value fills computed on the training split: median for numeric, a sentinel for categorical."""
    return {
        name: (0.0 if pd.isna(df[name].median()) else df[name].median()) if is_numeric(df[name]) else "(missing)"
        for name in features
    }


def _encode(df: pd.DataFrame, features: list[str], fill: dict) -> dict[str, np.ndarray]:
    """Model inputs keyed feature_<i> (column names may not be valid layer names)."""
    encoded = {}
    for i, name in enumerate(features):
        column = df[name].fillna(fill[name])
        values = column.to_numpy("float32") if is_numeric(df[name]) else column.astype(str).to_numpy(object)
        encoded[f"feature_{i}"] = values.reshape(-1, 1)
    return encoded


def _batches(x: dict[str, np.ndarray], y: np.ndarray, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    """tf.data rather than raw numpy: Keras 3 rejects numpy string arrays as fit/predict inputs."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(len(y), seed=0)
    return dataset.batch(batch_size)


class _Stream(keras.callbacks.Callback):
    def __init__(self, on_epoch: Callable[[dict], None], should_stop: Callable[[], bool]):
        super().__init__()
        self.on_epoch = on_epoch
        self.should_stop = should_stop

    def on_epoch_end(self, epoch, logs=None):
        self.on_epoch(jsonable({"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}}))

    def on_train_batch_end(self, batch, logs=None):
        if self.should_stop():
            self.model.stop_training = True


def train(
    config: RunConfig,
    df: pd.DataFrame,
    model_path: Path,
    on_epoch: Callable[[dict], None],
    should_stop: Callable[[], bool],
) -> dict:
    df = df[config.features + [config.label]].dropna(subset=[config.label])
    task = config.task or infer_task(df[config.label])

    classes = None
    if task == "classification":
        labels = df[config.label].astype(str)
        classes = sorted(labels.unique())
        if len(classes) < 2:
            raise ValueError("Classification needs at least 2 distinct label values.")
        y = pd.Categorical(labels, categories=classes).codes.astype("int32")
    else:
        if not is_numeric(df[config.label]):
            raise ValueError(f"Regression needs a numeric label; '{config.label}' is categorical.")
        y = df[config.label].to_numpy("float32")

    order = np.random.default_rng(0).permutation(len(df))
    n_val = int(len(df) * config.validation_split)
    val_idx, train_idx = order[:n_val], order[n_val:]
    if n_val == 0 or len(train_idx) == 0:
        raise ValueError("Not enough rows to make both a training and a validation split.")

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    fill = _fill_values(train_df, config.features)
    train_x, val_x = _encode(train_df, config.features, fill), _encode(val_df, config.features, fill)
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_model(config, train_x, len(classes) if classes else None)
    val_batches = _batches(val_x, y_val, EVAL_BATCH_SIZE)
    model.fit(
        _batches(train_x, y_train, config.batch_size, shuffle=True),
        validation_data=val_batches,
        epochs=config.epochs,
        callbacks=[_Stream(on_epoch, should_stop)],
        verbose=0,
    )
    model.save(model_path)

    metrics = model.evaluate(val_batches, return_dict=True, verbose=0)
    raw = model.predict(val_batches, verbose=0)
    if classes:
        predicted = [classes[i] for i in raw.argmax(axis=1)]
        observed = [classes[i] for i in y_val]
    else:
        predicted = raw[:, 0].tolist()
        observed = y_val.tolist()
        metrics["r2"] = 1 - np.sum((y_val - raw[:, 0]) ** 2) / np.sum((y_val - y_val.mean()) ** 2)

    shown = val_df.head(PREDICTION_ROWS)
    return jsonable({
        "task": task,
        "classes": classes,
        "metrics": metrics,
        "predictions": {
            "features": {name: shown[name].tolist() for name in config.features},
            "observed": observed[:PREDICTION_ROWS],
            "predicted": predicted[:PREDICTION_ROWS],
        },
    })
