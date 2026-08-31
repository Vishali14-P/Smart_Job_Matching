import pandas as pd

jobs = pd.read_csv("data/raw/jobs.csv")

print(jobs.head())
print("\nShape:", jobs.shape)

print("\nColumns:")
print(jobs.columns)

print("\nMissing values:")
print(jobs.isnull().sum())