import unittest
from fknn import *
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split



class TestGfmmMethods(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("iris-setosa.csv")
        self.X = self.df.iloc[:, 1:3].values
        self.y = self.df.iloc[:,0].values
        self.model = FuzzyKNN()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def test_isFitted(self):            # Controlla che la fit sia eseguita
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(self.model.fitted_)

    def test_isScoreValid(self):        # Controlla che l'accuratezza sia un valore valido
        warnings.filterwarnings("ignore")
        self.model.fit(self.X_train, self.y_train)
        acc = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_errorThreshold(self):       # Controlla che l'errore RMSE delle membership sia minore di una certa soglia
        warnings.filterwarnings("ignore")
        threshold = 0.3
        self.model.fit(self.X_train, self.y_train)
        err = self.model.RMSE_membership(self.X_test, self.y_test)
        self.assertLessEqual(err, threshold)

if __name__ == '__main__':
    unittest.main()
