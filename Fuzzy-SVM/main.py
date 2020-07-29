import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, RegressorMixin
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn import svm
import math
import itertools
import pylab as pl
import sys

from HYP_SVM import *

def hyp_svm(best_C, best_sigma):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        typ = 2
        best_model = HYP_SVM(C=best_C, kernel=gaussian_kernel, sigma=best_sigma)

        best_model.fit(X_train, y_train)
        best_model.score(X_test, y_test)

    return best_model.score(X_test, y_test)

if __name__=="__main__":
    data=sys.argv[1]
    dataset=pd.read_csv(data)

    X = dataset.columns[1:3]
    X = dataset[X]
    y = dataset.columns[0]
    y = dataset[y]

    X = np.array(X)
    y = np.array(y)
    X=X.astype(float)
    y=y.astype(float)
    y=np.where(y==0,-1,y)

    C_vals = [1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4]
    sigma = [9e-2, 9e-1, 9, 9e+1, 9e+2, 9e+3, 9e+4]
    #kernels = ["linear_kernel", "polynomial_kernel", "gaussian_kernel"]
    parameters = {'C': C_vals, 'sigma': sigma}

    model = HYP_SVM(C=100, kernel=gaussian_kernel, sigma=0.9)

    xTrain, xTest, yTrain, yTest = train_test_split(X, y)

    clf = GridSearchCV(model, parameters, cv=5)

    grid_result = clf.fit(X=xTrain, y=yTrain)

    clf.score(xTest, yTest)

    best_params = clf.best_params_
    best_C = best_params['C']
    best_sigma = best_params['sigma']
    print(best_C, best_sigma)
    #hyp_svm(best_C, best_sigma)
