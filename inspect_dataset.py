import pandas as pd

df = pd.read_csv("data/metadata.csv")

print(df.head())

print("\nNumber of samples:", len(df))

print("\nFrame lengths:")
print(df["frames"])

print("\nFeature count:")
print(df["features"].unique())