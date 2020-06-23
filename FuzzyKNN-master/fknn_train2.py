import sys

from fknn import *
from sklearn.model_selection import train_test_split
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

def fknn_train (filename):

	dataset = pd.read_csv(filename)

	model = FuzzyKNN()
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
	X = dataset.iloc[:, 1:3]
	Y = dataset.iloc[:0]

	X = np.array(X)
	Y = np.array(Y)
	print(X)

	value_array = []
	error_array = []

	for train_index, test_index in skf.split(X,Y):
		print("TRAIN:", train_index, "TEST:", test_index)
		xTrain, xTest = X[train_index], X[test_index]
		yTrain, yTest = Y[train_index], Y[test_index]
		model.fit(xTrain, yTrain)
		value = model.score(xTest, yTest)
		error = model.mean_squared_error(xTest, yTest)
		value_array.append(value)
		error_array.append(error)

	return model, value_array, error_array


def main():
    filename = sys.argv[-1]
    model, score, error = fknn_train (filename)
    print(score)

if __name__ == "__main__":
    main()
