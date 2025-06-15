import json
import pandas as pd
import os

schemas = {
    "AccountsAPI": "data/account-info-openapi.json",
    "PaymentsAPI": "data/payment-initiation-openapi.json",
    "FundsConfirmationAPI": "data/confirmation-funds-openapi.json",
    "VRPAPI": "data/vrp-openapi.json"
}

rows = []
for section, filepath in schemas.items():
    with open(filepath, "r", encoding="utf-8") as f:
        response = json.load(f)
        for name, details in response.get("components", {}).get("schemas", {}).items():
            for field, meta in details.get("properties", {}).items():
                rows.append({
                    "API Section": section,
                    "Object": name,
                    "Field": field,
                    "Type": meta.get("type", meta.get("$ref", "").split("/")[-1]),
                    "Description": meta.get("description", "")
                })

df = pd.DataFrame(rows)
output_path = "openbanking_uk_all_fields.csv"
df.to_csv(output_path, index=False)

print(f"CSV generated successfully at {output_path}!")
