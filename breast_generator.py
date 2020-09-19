
import pandas as pd

df = pd.read_csv ("Dataset/breast-cancer.csv")

colname = "class"
cols = list(df.columns.values)
cols.pop(cols.index(colname))
df = df[[colname]+cols]

df.replace(to_replace="recurrence-events", value=1, inplace=True)
df.replace(to_replace='false-recurrence-events', value=0, inplace=True)

df.replace(to_replace='ge40', value=0, inplace=True)
df.replace(to_replace='premefalse', value=1, inplace=True)
df.replace(to_replace='lt40', value=2, inplace=True)

df.replace(to_replace='True', value=1, inplace=True)
df.replace(to_replace='False', value=0, inplace=True)

df.replace(to_replace='left', value=1, inplace=True)
df.replace(to_replace='right', value=1, inplace=True)

df.replace(to_replace='left_up', value=0, inplace=True)
df.replace(to_replace='central', value=1, inplace=True)
df.replace(to_replace='left_low', value=2, inplace=True)
df.replace(to_replace='right_up', value=3, inplace=True)
df.replace(to_replace='right_low', value=4, inplace=True)

df.to_csv('Dataset/breast-std.csv', index=False)
