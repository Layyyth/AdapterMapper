import pandas as pd
import json


df = pd.read_csv("data\download.csv")

missing_desc_df = df[df['description'].isnull() | (df['description'].str.strip() == '')]


missing_fields = missing_desc_df['field_name'].tolist()
with open("missing_description_fields.json", "w", encoding="utf-8") as f:
    json.dump(missing_fields, f, indent=2, ensure_ascii=False)

print(f" Extracted {len(missing_fields)} fields with missing descriptions.")
print(" Output saved to 'missing_description_fields.json'")
