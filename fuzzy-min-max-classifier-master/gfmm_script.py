import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle
import sys
from fuzzy import *
import logging



def gfmm_train_test(filename):
    df = pd.read_csv(filename)

    X = df.iloc[:, 1:3].values
    y = df.iloc[:,0].values

    seed = 10
    X, y = shuffle(X, y, random_state=seed)

    a = np.arange (0.0, 1.1, 0.1)
    parameters = {'sensitivity': a, 'exp_bound': a }
    N_SPLIT = 5
    err = []
    acc = []


    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=False, random_state=5)
    for train_index, validation_index in skf.split(X, y):
        X_train, X_validation = X[train_index], X[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        model = FuzzyMMC(sensitivity=1, exp_bound=0.1, animate=False)
        clf = GridSearchCV(model, parameters, cv=5)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        best_model.fit(X_train, y_train)
        acc.append(best_model.score(X_validation, y_validation))
        val = best_model.RMSE_membership(X_validation, y_validation)
        err.append(val)
    return best_model, acc, err



def main():
    logging.basicConfig(filename = 'esperimenti.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    filename = sys.argv[-1]
    logging.info('Questo esperimento lavora sul dataset: %s', filename)
    model, score_array, error_array = gfmm_train_test (filename)
    logging.info(f'{score_array} = array delle accuratezze')
    logging.info(f'{error_array} = array delle RMSE_membership \n')
    print(score_array)
    print(error_array)

if __name__ == "__main__":
    main()
