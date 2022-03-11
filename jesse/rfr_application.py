# Setup
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from random_forest import RandomForestRegressor

# Config variables
training_data_path = '../datasets/bpi_2012_train_eng.csv'
testing_data_path = '../datasets/bpi_2012_test_eng.csv'

n_samples = 1000
n_trees = 3
sample_split = 5
max_depth = 5

# Loading and splitting the datasets
df_train = pd.read_csv(training_data_path)
df_train = df_train.set_index('event_index').drop('Unnamed: 0', axis=1)

df_test = pd.read_csv(testing_data_path)
df_test = df_test.set_index('event_index').drop('Unnamed: 0', axis=1)

# Selecting columns and rows
df_train = df_train.drop(['nextEvent', 'nextEventTime'], axis=1)[0:n_samples]
df_test = df_test.drop(['nextEvent', 'nextEventTime'], axis=1)[0:n_samples]

df_train = df_train.dropna()
df_test = df_test.dropna()


# Starting time
start_time = time.time()

X_train = df_train.drop(['nextEventTimeRel'], axis=1).values
Y_train = df_train['nextEventTimeRel'].values.reshape(-1, 1)
X_test = df_test.drop(['nextEventTimeRel'], axis=1).values
Y_test = df_test['nextEventTimeRel'].values.reshape(-1, 1)

# Constructing and fitting the model
regressor = RandomForestRegressor(n_trees=n_trees, min_samples_split=sample_split, max_depth=max_depth)
regressor.fit(X_train, Y_train)

# Predicting the values of our test dataset
Y_pred = regressor.predict(X_test)
msq = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Retrieving the accuracy of the model
print(f'Accuracy score: {msq}')

# Ending time
end_time = time.time()
print(f'\r\nThe execution of Random Forest took {round(end_time - start_time)} seconds')
