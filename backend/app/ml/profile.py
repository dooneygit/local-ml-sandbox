import math

import pandas as pd

SCATTER_SAMPLE_ROWS = 2000


def jsonable(value):
    """Recursively replace NaN/inf (invalid JSON) with None and numpy scalars with Python ones."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def profile_dataset(df: pd.DataFrame) -> dict:
    columns = []
    for name in df.columns:
        series = df[name]
        column = {
            "name": name,
            "kind": "numeric" if is_numeric(series) else "categorical",
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
        }
        if column["kind"] == "numeric":
            column |= {stat: series.agg(stat) for stat in ("mean", "std", "min", "max")}
        columns.append(column)

    numeric = df.select_dtypes("number")
    corr = numeric.corr()
    sample = numeric.sample(min(len(numeric), SCATTER_SAMPLE_ROWS), random_state=0)

    return jsonable({
        "n_rows": len(df),
        "columns": columns,
        "correlation": {"columns": list(corr.columns), "values": corr.values.tolist()},
        "scatter_sample": {name: sample[name].tolist() for name in sample.columns},
    })
