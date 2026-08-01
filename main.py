import numpy as np
import pandas as pd
import keras
import plotly.express as px
from experiment import Hyperparameters, Experiment
from model import create_model, train_model
from plot import plot_experiment_metrics, plot_model_predictions


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

hyperparameters_1 = Hyperparameters(
    input_features=["TRIP_MILES"],
    learning_rate=0.001,
    batch_size=50,
    number_epochs=20
)

metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]

model_1=create_model(hyperparameters_1, metrics)

experiment_1 = train_model('one_feature_test', model_1, training_df, 'FARE', hyperparameters_1)

plot_experiment_metrics(experiment_1, ["rmse"])
plot_model_predictions(experiment_1, training_df, "FARE")
