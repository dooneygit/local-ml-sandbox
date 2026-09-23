import numpy as np
import pandas as pd
import keras
import plotly.express as px
from experiment import Hyperparameters, Experiment
from model import create_model, train_model
from plot import plot_experiment_metrics, plot_model_predictions
from prediction import predict_fare, show_predictions


taxi_dataset = pd.read_csv("taxi_train.csv")

training_df = taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
print(f'Total rows: {len(training_df.index)}\n')

print('Dataset Statistics')
print(f"{training_df.describe(include='all')}\n")

print('Correlation Matrix')
print(training_df.corr(numeric_only= True))

pair_plot = px.scatter_matrix(
    training_df, 
    dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"],
)

pair_plot.show()

training_df["TRIP_MINUTES"] = training_df["TRIP_SECONDS"]/60

hyperparameters = Hyperparameters(
    input_features=["TRIP_MILES", "TRIP_SECONDS"],
    learning_rate=0.001,
    batch_size=50,
    number_epochs=20
)

metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]

model=create_model(hyperparameters, metrics)

experiment_1 = train_model('test', model, training_df, 'FARE', hyperparameters)

plot_experiment_metrics(experiment_1, ["rmse"])
plot_model_predictions(experiment_1, training_df, "FARE")
