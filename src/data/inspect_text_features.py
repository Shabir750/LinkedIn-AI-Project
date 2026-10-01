import pandas as pd

FILE = "data/linkedin_ai_dataset/text_features.csv"

df = pd.read_csv(FILE)

print("Total columns:", len(df.columns))

print("\nFirst 10 columns:")
for i, column in enumerate(df.columns[:10]):
    print(i, column)

print("\nLast 10 columns:")
for i, column in enumerate(df.columns[-10:], start=len(df.columns) - 10):
    print(i, column)