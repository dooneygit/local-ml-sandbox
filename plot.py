import numpy as np
import pandas as pd
import plotly.express as px


def plot_experiment_metrics(experiment, metric_names: list[str]) -> None:
    history = experiment.metrics_history.copy()
    history["epoch"] = experiment.epochs

    figure = px.line(
        history,
        x="epoch",
        y=metric_names,
        title=f"Training metrics: {experiment.name}",
    )
    figure.show()


def plot_model_predictions(
    experiment,
    dataset: pd.DataFrame,
    label_name: str,
) -> None:
    feature_name = experiment.settings.input_features[0]

    ordered_data = dataset.sort_values(feature_name)
    features = {
        name: ordered_data[name].to_numpy()
        for name in experiment.settings.input_features
    }

    predictions = experiment.model.predict(features, verbose=0).flatten()

    figure = px.scatter(
        ordered_data,
        x=feature_name,
        y=label_name,
        title=f"Model predictions: {experiment.name}",
    )

    figure.add_scatter(
        x=ordered_data[feature_name],
        y=predictions,
        mode="lines",
        name="Prediction",
    )

    figure.show()