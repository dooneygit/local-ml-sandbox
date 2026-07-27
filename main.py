import numpy as np
import pandas as pd
import keras
import ml_edu.experiment
import ml_edu.results
import plotly.express as px

taxi_dataset = pd.read_csv("taxi_train.csv")

training_df = taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
print(f'Total rows: {len(training_df.index)}\n')

print('Dataset Statistics')
print(f'{training_df.describe(include='all')}\n')

print('Correlation Matrix')
print(training_df.corr(numeric_only= True))