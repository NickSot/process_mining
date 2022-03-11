import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

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
            dt.print_tree()
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


class RandomForestRegressor():
    """
    A class used to represent a Random Forest for Regression
    """

    def __init__(self, n_trees: int, min_samples_split: int = 2, max_depth: int = 2):

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
            df_sample = df.sample(n=len(df), replace=True,random_state=np.random.RandomState())

            print(f'{i}: Sampling complete!')

            # Random feature selection

            # Fit a new tree with the sampled dataset
            X_sample, Y_sample = df_sample.iloc[:,:-1], pd.DataFrame(df_sample.iloc[:, -1])
            dt = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            dt.fit(X_sample, Y_sample)
            print(f'{i}: Training complete!')

            # Store the tree in the forest
            forest.append(dt)
            dt.print_tree()

        self.forest = forest

    def predict(self, X):
        results = []

        # Feed the dataset through every tree and store the results in a list
        for index, dt in enumerate(self.forest):
            # Result is a list of the predicted value for every datapoint in the list
            result = dt.predict(X)
            results.append(result)

        # Turn the results into a dataframe. Every row is a datapoint, every column represents a different tree.
        df_results = pd.DataFrame(results)
        df_results = df_results.transpose()

        # Aggregation (take majority vote)
        df_vote_result = df_results.mean(axis=1)

        # The result is now a single column with the mean per datapoint.
        print('\r\nThe mean calculation concluded:')
        print(df_vote_result.head())
        return df_vote_result



runFromFile = False

if runFromFile:
    # Setup
    import numpy as np
    import pandas as pd
    import time
    from sklearn.metrics import mean_squared_error
    from random_forest import RandomForestRegressor

    # Config variables
    training_data_path = '../datasets/bpi_2012_train_eng.csv'
    testing_data_path = '../datasets/bpi_2012_test_eng.csv'

    n_samples = 2000
    n_trees = 3
    sample_split = 2
    max_depth = 2

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
