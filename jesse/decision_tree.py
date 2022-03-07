import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        '''Constructor'''

        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # For leaf node
        self.value = value

class DecisionTreeClassifier():
    """
    A class used to represent a Decision Tree for Classification

    Methods
    -------
    build_tree(self, dataset, curr_depth=0)
        Builds the Decision Tree using recursion
    """

    def __init__(self, min_samples_split=2, max_depth=2):
        '''Constructor'''

        # Initialize the root of the tree
        self.root = None

        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    def build_tree(self, dataset : pd.DataFrame, curr_depth : int =0) -> Node:
        '''Recursive method to build the tree

        Parameters
        ----------
        dataset : `Dataframe`
            The dataset to build the tree on
        curr_depth : `int`
            The current depth of the tree under construction

        Returns
        -------
        `Node`
            Either a Decision Node or Leaf Node
        '''

        X, Y = dataset[:, :-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # Split until the stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            
            # Find the best split
            best_split = self.get_best_split(dataset, num_features)

            if best_split['info_gain'] > 0:
                # Recurse left
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth + 1)

                # Recurse right
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth + 1)

                # Return the decision node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])
        
        # Compute leaf node
        leaf_value = self.calculate_leaf_value(Y)

        # Return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset : pd.DataFrame, num_features : int) -> dict:
        '''Method to find the best split
        
        This is a greedy method that finds the best feature and value to split\\
        on to maximize information gain. It does not guarantee an optimal\\
        decision tree, but for its simplicity it performs quite well.

        Parameters
        ----------
        dataset : `Dataframe`
            The dataset to find the best split on
        num_samples : `int`
            The amount of samples in the given dataset
        num_features : `int`
            The amount of features in the given dataset
        
        Returns
        -------
        `Dictionary`
            A dictionary representing the best split, of the following form:\\
            {
                feature_index : `int`,
                threshold : `int`,
                dataset_left : `Dataframe`,
                dataset_right: `Dataframe`,
                info_gain : `float`
            }
        '''
        # Dictionary to store the best split
        best_split = {}
        max_info_gain = -float('inf')

        # Loop over all features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # Check all possible threshold values to find the best
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, 'gini')
                    if curr_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain
        
        # Return the best split
        return best_split

    def split(self, dataset : pd.DataFrame, feature_index : int, threshold : int) -> tuple:
        '''Method to split the dataset
        
        This method splits the dataset on the given feature index and threshold.\\
        It takes every item in the dataset and puts it either in the left dataset\\
        if it's smaller than the threshold, or in the right dataset if it's larger\\
        than the threshold.

        Parameters
        ----------
        dataset : `Dataframe`
            The dataset to perform the split on
        feature_index : `int`
            The feature on which to split the dataset
        threshold : `int`
            The threshold value on which to split the dataset
        
        Returns
        -------
        `tuple(Dataframe, Dataframe)`
            A tuple representing the left and right datasets respectively
        '''
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        """Method to compute the information gain"""

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode == 'gini':
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        
        return gain
    
    def entropy(self, y):
        ''' Method to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        '''Method to calculate the value of the leaf `Node`'''

        Y = list(Y)
        return max(Y, key=Y.count)

    def fit(self, X, Y):
        '''Method to train the tree'''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        '''Method to make predictions on a new dataset'''

        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        """Method to predict a single data point using the given tree"""
        
        # If the given node (tree) is a leaf node, return it as the prediction
        if tree.value != None : return tree.value

        # If it's not a leaf node, decide whether the datapoint should be on the
        # left or on the right and recurse
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


    # Util
    def print_tree(self, tree=None, indent=" "):
        '''Method to print the built tree'''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
