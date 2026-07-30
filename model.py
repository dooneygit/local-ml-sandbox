import keras
from hyperparameters import Hyperparameters

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