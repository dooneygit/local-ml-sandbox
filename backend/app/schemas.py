from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

Task = Literal["regression", "classification"]


class RunConfig(BaseModel):
    dataset_id: str
    label: str
    features: list[str] = Field(min_length=1)
    task: Task | None = None  # None = infer from the label column
    name: str | None = None
    learning_rate: float = Field(0.001, gt=0)
    batch_size: int = Field(50, ge=1)
    epochs: int = Field(20, ge=1, le=1000)
    hidden_layers: list[PositiveInt] = Field(default_factory=list)  # [] = linear / logistic model
    optimizer: Literal["rmsprop", "adam", "sgd"] = "rmsprop"
    validation_split: float = Field(0.2, gt=0, lt=1)
