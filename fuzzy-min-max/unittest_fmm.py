import unittest
from fuzzy import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TestGfmmMethods(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("iris-setosa.csv")
        self.X = self.df.iloc[:, 1:3].values
        self.y = self.df.iloc[:,0].values
        self.model = FuzzyMMC(sensitivity=1, exp_bound=0.1, animate=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def test_isFitted(self):            # Controlla che la fit sia eseguita
        self.model.fit(self.X_train, self.y_train)
        self.assertTrue(self.model.fitted_)

    def test_isScoreValid(self):        # Controlla che l'accuratezza sia un valore valido
        self.model.fit(self.X_train, self.y_train)
        acc = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_errorThreshold(self):       # Controlla che l'errore RMSE delle membership sia minore di una certa soglia
        threshold = 0.7
        self.model.fit(self.X_train, self.y_train)
        err = self.model.RMSE_membership(self.X_test, self.y_test)
        print(err)
        self.assertLessEqual(err, threshold)

    def test_isLabelValid (self):       # Controlla che le label predette siano o 0 o 1
        self.model.fit(self.X_train, self.y_train)
        _, _, lab = self.model.predict(self.X_test[0])
        self.assertRegex(str(lab), '[0-1]')

if __name__ == '__main__':
    unittest.main()
