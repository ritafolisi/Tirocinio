
import pandas as pd

df = pd.read_csv ("iris.csv")

colname = "species"
cols = list(df.columns.values)
cols.pop(cols.index(colname))
df = df[[colname]+cols]
df1=df.copy()
df2=df.copy()

df.replace(to_replace="setosa", value=1, inplace=True)
df.replace(to_replace='virginica', value=0, inplace=True)
df.replace(to_replace='versicolor', value=0, inplace=True)

df.to_csv('iris-setosa.csv', index=False)

df = pd.read_csv ("iris.csv")

df1.replace(to_replace="setosa", value=0, inplace=True)
df1.replace(to_replace='virginica', value=1, inplace=True)
df1.replace(to_replace='versicolor', value=0, inplace=True)

df1.to_csv('iris-virginica.csv', index=False)

df2.replace(to_replace="setosa", value=0, inplace=True)
df2.replace(to_replace='virginica', value=0, inplace=True)
df2.replace(to_replace='versicolor', value=1, inplace=True)

df2.to_csv('iris-versicolor.csv', index=False)
