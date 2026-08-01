import keras
from experiment import Hyperparameters, Experiment
import pandas as pd

def create_model(
    hyperparameters: Hyperparameters,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    inputs = {name: keras.Input(shape=(1,), name=name) for name in hyperparameters.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs= keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=hyperparameters.learning_rate),
        loss = "mean_squared_error",
        metrics=metrics
    )

    return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: Hyperparameters
) -> Experiment:
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(x=features,
                        y=label,
                        batch_size=Hyperparameters.batch_size,
                        epochs=Hyperparameters.number_epochs)

    return Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epochs,
        metrics_history=pd.DataFrame(history.history),
    )

    