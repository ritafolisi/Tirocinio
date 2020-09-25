
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv ("Dataset/breast-cancer.csv")

# Posiziono la classe come prima colonna
colname = "class"
cols = list(df.columns.values)
cols.pop(cols.index(colname))
df = df[[colname]+cols]

# Sostituisco gli intervalli con il valore medio
colnames = ['age', 'tumor-size', 'inv-falsedes']
for colname in colnames:
    conta=0
    for i in df[colname]:
        x = i.split("-")
        df.at[conta, colname] = (int(x[0]) + int (x[1]))/2
        conta +=1


colnames = ['class', 'mefalsepause', 'falsede-caps', 'breast', 'irradiat',  ]
for colname in colnames:
    le = preprocessing.LabelEncoder()
    target=le.fit_transform(df[colname])

    df[colname]=target

df.replace(to_replace='left_up', value=0, inplace=True)
df.replace(to_replace='central', value=1, inplace=True)
df.replace(to_replace='left_low', value=2, inplace=True)
df.replace(to_replace='right_up', value=3, inplace=True)
df.replace(to_replace='right_low', value=4, inplace=True)

df.to_csv('Dataset/breast-std.csv', index=False)
