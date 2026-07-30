from dataclasses import dataclass

@dataclass
class Hyperparameters:
    input_features: list[str]
    learning_rate: float
    batch_size = int
    number_epochs = int