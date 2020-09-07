import sys

from fknn import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging



def fknn_script (filename):

	df = pd.read_csv(filename)

	# Inizializzazione variabili
	X = df.iloc[:, 1:3].values
	y = df.iloc[:,0].values

	# Shuffle dataset
	seed = 10
	X, y = shuffle(X, y, random_state=seed)

	a = np.arange (1, 21, 2)
	parameters = {"k" : a}
	N_SPLIT = 5
	err = []
	acc = []

	# Tuning parametri e cross validation
	skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=False, random_state=5)
	for train_index, validation_index in skf.split(X, y):
		print(train_index)
		X_train, X_validation = X[train_index], X[validation_index]
		y_train, y_validation = y[train_index], y[validation_index]

		model = FuzzyKNN()
		clf = GridSearchCV(model, parameters, cv=5)
		clf.fit(X_train, y_train)
		best_model = clf.best_estimator_
		best_model.fit(X_train, y_train)
		acc.append(best_model.score(X_validation, y_validation))
		val = best_model.RMSE_membership(X_validation, y_validation)
		err.append(val)
	return model, acc, err


def main():
	logging.basicConfig(filename = 'esperimenti.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
	filename = sys.argv[-1]
	logging.info('Questo esperimento lavora sul dataset: %s', filename)
	model, score, error = fknn_script (filename)
	logging.info(f'{score} = array delle accuratezze')
	logging.info(f'{error} = array delle RMSE_membership \n')
	print(score)
	print (error)

if __name__ == "__main__":
    main()
