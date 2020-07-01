import sys
from fuzzy import *
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV



def gfmm_train_test (filename):
  df = pd.read_csv(filename)

  X = df.iloc[:, 1:3].values
  Y = df.iloc[:,0].values

 # xTrain, xTest, yTrain, yTest = train_test_split(X,Y)

  model = FuzzyMMC(sensitivity=1, exp_bound=0.1, animate=False)
  a = np.arange (0.0, 1.1, 0.1)
  parameters = {'sensitivity': a, 'exp_bound': a }
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
  clf = GridSearchCV(model, parameters, cv = 5)
  value_array = []
  error_array = []

  xTrain, xTest, yTrain, yTest = train_test_split(X,Y)
  clf.fit(xTrain, yTrain)
  best_params = clf.best_params_
  new_exp = best_params['exp_bound']
  new_sensit = best_params['sensitivity']

  model = FuzzyMMC(sensitivity=new_sensit, exp_bound=new_exp, animate=False)

  for train_index, test_index in skf.split(X, Y):
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
    model, score_array, error_array = gfmm_train_test (filename)
    print(score_array)
    print(error_array)

if __name__ == "__main__":
    main()
