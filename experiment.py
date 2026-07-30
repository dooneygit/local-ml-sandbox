from dataclasses import dataclass
import keras
import pandas as pd

@dataclass
class Hyperparameters:
    input_features: list[str]
    learning_rate: float
    batch_size = int
    number_epochs = int

@dataclass
class Experiment:
    name: str
    settings: Hyperparameters
    model: keras.Model
    epochs: list[int]
    metrics_history: pd.DataFrame