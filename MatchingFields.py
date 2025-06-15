import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import re
from sklearn.metrics.pairwise import cosine_similarity


try:
    ob_df = pd.read_csv("data/UKOB_Fields.csv")
except FileNotFoundError:
    print("Error: Make sure 'uk_open_banking_fields_enriched_completed.csv' is in the correct directory.")
    exit()


def clean_ob_fieldname(name):
    return name.split('.')[-1]


abbreviation_map = {
    "acc": "account", "acct": "account", "usr": "user", "id": "identifier",
    "nm": "name", "cnst": "consent", "amt": "amount", "txn": "transaction",
    "num": "number", "ref": "reference", "cred": "creditor", "deb": "debtor",
    "auth": "authorisation", "curr": "currency", "dt": "date", "exp": "expiration",
    "exch": "exchange", "multi": "multi", "proxy": "proxy", "addr": "address",
    "post": "postal", "prod": "product", "issuer": "issuer", "status": "status",
    "upd": "update", "info": "information", "ben": "beneficiary", "pmt": "payment",
    "init": "initiation", "trf": "transfer", "ind": "indicator", "val": "value", "iban": "iban"
}

def preprocess_bank_field_name(name):
    parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).replace("_", " ").lower().split()
    expanded_parts = [abbreviation_map.get(part, part) for part in parts]
    return " ".join(expanded_parts)


ob_texts = (
    ob_df['field_name'].apply(clean_ob_fieldname) + ": " + ob_df['description'].fillna("")
).tolist()


model = SentenceTransformer('all-MiniLM-L12-v2')
ob_embeddings = model.encode(ob_texts, convert_to_numpy=True)
embedding_dim = ob_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(ob_embeddings)


with open("data/bank_sample.json", "r", encoding="utf-8") as file:
    bank_field_names = json.load(file)


high_confidence_threshold = 0.85
medium_confidence_threshold = 0.70
min_confidence_gap = 0.1

mappings = []
mismatches = []
auto_mapped_count = 0

# === Match loop ===
for name in bank_field_names:
    cleaned_name = preprocess_bank_field_name(name)
    query_embedding = model.encode([cleaned_name], convert_to_numpy=True)

    D, I = index.search(query_embedding, k=5) 
    top_indices = I[0]
    candidate_embeddings = ob_embeddings[top_indices]

    cosine_scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
    sorted_indices = np.argsort(cosine_scores)[::-1]

    best_idx = top_indices[sorted_indices[0]]
    top_score = cosine_scores[sorted_indices[0]]

    second_best_score = -1.0
    if len(sorted_indices) > 1:
        second_best_score = cosine_scores[sorted_indices[1]]
    confidence_gap = top_score - second_best_score

    match_name = ob_df.iloc[best_idx]['field_name']
    match_desc = ob_df.iloc[best_idx]['description']

    print(f"\n🔍 Bank Field: {name}")
    print(f"   Preprocessed: '{cleaned_name}'")

    if top_score >= high_confidence_threshold or \
       (top_score >= medium_confidence_threshold and confidence_gap >= min_confidence_gap):
        print(f"✅ Auto-Mapped to: {match_name} (Score: {top_score:.2f}, Gap: {confidence_gap:.2f})")
        mappings.append({
            "bank_field": name, "preprocessed": cleaned_name, "mapped_to": match_name,
            "description": match_desc, "confidence": float(round(top_score, 2))
        })
        auto_mapped_count += 1
    else:
        print(f"⚠️ Low Confidence (Score: {top_score:.2f}, Gap: {confidence_gap:.2f}) — needs manual review")
        print("Top 3 suggestions:")
        top_suggestions = []
        for rank, rel_idx in enumerate(sorted_indices[:3]):
            idx = top_indices[rel_idx]
            alt_name = ob_df.iloc[idx]['field_name']
            alt_desc = ob_df.iloc[idx]['description']
            score = cosine_scores[rel_idx]
            print(f"   {rank+1}. {alt_name} (Score: {score:.2f})")
            top_suggestions.append({
                "suggestion": alt_name, "description": alt_desc, "confidence": float(round(score, 2))
            })
        mismatches.append({
            "bank_field": name, "preprocessed": cleaned_name, "top_suggestions": top_suggestions
        })


with open("field_mappings_improved.json", "w", encoding="utf-8") as out:
    json.dump(mappings, out, indent=2, ensure_ascii=False)
with open("mismatches_log_improved.json", "w", encoding="utf-8") as out:
    json.dump(mismatches, out, indent=2, ensure_ascii=False)

total = len(bank_field_names)
print(f"\n✅ Auto-mapped {auto_mapped_count}/{total} fields ({(auto_mapped_count/total)*100:.2f}%)")
print("📄 Mappings saved to 'field_mappings_improved.json'")
print("📄 Mismatches saved to 'mismatches_log_improved.json'")