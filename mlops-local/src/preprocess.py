import pandas as pd

df = pd.read_csv("data/raw.csv")
# drop all categorical columns
df = df.select_dtypes(include=["number"])
# convert all na values to mean
df = df.fillna(df.mean())
df.to_csv("data/preprocessed.csv", index=False)
