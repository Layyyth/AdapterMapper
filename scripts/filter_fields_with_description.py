import pandas as pd


df = pd.read_csv("data/download.csv")

df_cleaned = df[df['description'].notna() & df['description'].str.strip().ne('')]


df_cleaned.to_csv("filtered_output.csv", index=False)

print(f" Kept {len(df_cleaned)} rows with valid descriptions.")
