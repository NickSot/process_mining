import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier
from collections import Counter

class RandomForestClassifier():
    """
    A class used to represent a Random Forest for Classification
    """
    
    def __init__(self, n_trees : int, min_samples_split : int = 2, max_depth : int = 2):

        # Initialize the forest
        self.forest = None

        # Stopping conditions
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, Y):
        '''Method to train the Random Forest'''

        forest = []
        
        for i in range(self.n_trees):

            print(f'\r\n{i}: Building tree...')

            # Bootstrapping (random sampling with replacement)
            dataset = np.concatenate((X, Y), axis=1)
            df = pd.DataFrame(dataset)
            df_sample = df.sample(n=len(df), replace=True, random_state=np.random.RandomState())

            print(f'{i}: Sampling complete!')

            # Random feature selection

            # Fit a new tree with the sampled dataset
            X_sample, Y_sample = df_sample.iloc[:, :-1], pd.DataFrame(df_sample.iloc[:, -1])
            dt = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            dt.fit(X_sample, Y_sample)
            print(f'{i}: Training complete!')

            # Store the tree in the forest
            forest.append(dt)
        
        self.forest = forest
    
    def predict(self, X):
        results = []

        # Feed the dataset through every tree and store the results in a list
        for index, dt in enumerate(self.forest):
            result = dt.predict(X) # Result is a list of the predicted value for every datapoint in the list
            results.append(result)
        
        # Turn the results into a dataframe. Every row is a datapoint, every column represents a different tree.
        df_results = pd.DataFrame(results)
        df_results = df_results.transpose()

        # Aggregation (take majority vote)
        df_vote_result = df_results.mode(axis=1)[[0]]

        # The result is now a single column with the majority vote per datapoint.

        print('\r\nThe majority vote concluded:')
        print(df_vote_result.head())
        return df_vote_result





# Running the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Starting time
start_time = time.time()

# Loading and splitting the datasets
n_samples = 5000

df_train = pd.read_csv('C:/Users/20203477/Documents/GitHub/process_mining/bpi_2012_train.csv')
df_train = df_train[['eventID ', 'case concept:name', 'case REG_DATE', 'event lifecycle:transition', 'event time:timestamp', 'event concept:name']].set_index('eventID ')[0:n_samples]

X = df_train.iloc[:, :-1].values # The original values without the target variable
Y = df_train.iloc[:, -1].values.reshape(-1, 1) # Turn the target variable into a single-column array of arrays

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

# Constructing and fitting the model
classifier = RandomForestClassifier(n_trees=10, min_samples_split=24, max_depth=24)
classifier.fit(X_train, Y_train)

# Predicting the values of our test dataset
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Retrieving the accuracy of the model
print(f'Accuracy score: {accuracy}')

# Ending time
end_time = time.time()
print(f'\r\nThe execution of Random Forest took {round(end_time - start_time)} seconds')