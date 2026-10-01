import pandas as pd

FILE = "data/linkedin_ai_dataset/train.csv"

df = pd.read_csv(FILE)

print("Total columns:", len(df.columns))

print("\nFirst 20 columns:")
for i, column in enumerate(df.columns[:20]):
    print(i, column)

print("\nLast 10 columns:")
for i, column in enumerate(df.columns[-10:], start=len(df.columns)-10):
    print(i, column)