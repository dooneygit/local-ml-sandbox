import keras
import numpy as np

from app.schemas import RunConfig

MAX_CATEGORIES = 100  # one-hot width cap for categorical features (most frequent kept, rest -> OOV)

OPTIMIZERS = {
    "rmsprop": keras.optimizers.RMSprop,
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
}


def create_model(config: RunConfig, train_x: dict[str, np.ndarray], n_classes: int | None) -> keras.Model:
    """Dense net over normalized numeric / one-hot categorical inputs. n_classes=None means regression.

    Preprocessing layers are adapted on the training split, so the saved model predicts from raw values.
    """
    inputs, encoded = {}, []
    for name, values in train_x.items():
        if values.dtype == object:
            inputs[name] = keras.Input(shape=(1,), dtype="string", name=name)
            preprocess = keras.layers.StringLookup(max_tokens=MAX_CATEGORIES, output_mode="one_hot")
        else:
            inputs[name] = keras.Input(shape=(1,), name=name)
            preprocess = keras.layers.Normalization()
        preprocess.adapt(values)
        encoded.append(preprocess(inputs[name]))

    x = keras.layers.Concatenate()(encoded) if len(encoded) > 1 else encoded[0]
    for units in config.hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    if n_classes is None:
        outputs = keras.layers.Dense(1)(x)
        loss = "mean_squared_error"
        metrics = [keras.metrics.RootMeanSquaredError(name="rmse"), keras.metrics.MeanAbsoluteError(name="mae")]
    else:
        outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=OPTIMIZERS[config.optimizer](learning_rate=config.learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model
