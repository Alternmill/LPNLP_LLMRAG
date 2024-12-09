import pandas as pd

df = pd.read_csv('../../data/cooking_entries.csv')

print(df.head())
entry = df[0]
print(entry)