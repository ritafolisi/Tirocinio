import sys
from fuzzy import *
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split


def gfmm_train (filename):
  df = pd.read_csv(filename)
  df = df.sample(frac=1)

  X = df.iloc[:, 1:3].values
  Y = df.iloc[:,0].values


  xTrain, xTest, yTrain, yTest = train_test_split(X,Y)

  model = FuzzyMMC(sensitivity=1, exp_bound=0.1, animate=False)
  model.fit(xTrain, yTrain)
  value = model.score(xTest, yTest)

  return model, value

def main():
    filename = sys.argv[-1]
    model, score = gfmm_train (filename)
    print(score)

if __name__ == "__main__":
    main()
