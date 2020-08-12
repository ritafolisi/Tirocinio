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
from sklearn.model_selection import GridSearchCV

def fknn_script (filename):

	dataset = pd.read_csv(filename)

	model = FuzzyKNN()
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


    # Inizializzazione variabili
	X = dataset.iloc[:, 1:3].values
	Y = dataset.iloc[:, 0].values

	a = np.arange (1, 21, 2)
	parameters = {"k" : a}
	clf = GridSearchCV(model, parameters, cv = 5)

	value_array = []
	error_array = []


    # Tuning parametri
	xTrain, xTest, yTrain, yTest = train_test_split(X,Y)
	clf.fit(xTrain, yTrain)
	model = clf.best_estimator_

    # Cross validation
	for train_index, test_index in skf.split(X,Y):
		print("TRAIN:", train_index, "TEST:", test_index)
		xTrain, xTest = X[train_index], X[test_index]
		yTrain, yTest = Y[train_index], Y[test_index]
		model.fit(xTrain, yTrain)
		value = model.score(xTest, yTest)
		error = model.mean_squared_error(xTest, yTest)
		value_array.append(value)
		error_array.append(error)
               
	RMSE_memb = model.RMSE_membership(xTest, yTest)
	return model, RMSE_memb, value_array, error_array


def main():
    filename = sys.argv[-1]
    model, RMSE_memb, score, error = fknn_script (filename)
    print(RMSE_memb)

if __name__ == "__main__":
    main()
