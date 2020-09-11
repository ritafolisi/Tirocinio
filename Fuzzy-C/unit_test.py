import unittest
from FCM import *
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split


class TestFCM(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("iris-setosa.csv")
        self.X = self.df.iloc[:, 1:3].values
        self.y = self.df.iloc[:,0].values
        self.model = FCM()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def test_isScoreValid(self):        # Controlla che l'accuratezza sia un valore valido
        n_clusters=2
        train_membership, centers = self.model.fuzzy_train(self.X_train, n_clusters, 2)
        test_membership = self.model.fuzzy_predict(self.X_test , n_clusters , centers, 2)
        #warnings.filterwarnings("ignore")
        acc = self.model.accuracy(test_membership, self.y_test)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_errorThreshold(self):       # Controlla che l'errore RMSE delle membership sia minore di una certa soglia
        threshold = 0.4
        n_clusters=2
        train_membership, centers = self.model.fuzzy_train(self.X_train, n_clusters, 2)
        test_membership = self.model.fuzzy_predict(self.X_test , n_clusters , centers, 2)
        #warnings.filterwarnings("ignore")
        err = self.model.RMSE_membership(test_membership, self.y_test)
        self.assertLessEqual(err, threshold)

if __name__ == '__main__':
    unittest.main()
