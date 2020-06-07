from sklearn import preprocessing
import pandas as pd
import sys


data = pd.read_csv(str(sys.argv[1]), sep=',')
colname = str(sys.argv[2])
le = preprocessing.LabelEncoder()
target=le.fit_transform(data[colname])

data[colname]=target

cols = list(data.columns.values)
cols.pop(cols.index(colname))
data = data[[colname]+cols]

data.to_csv('iris_std.csv')
