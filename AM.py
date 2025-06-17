import json
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

ob_json_path = "data/OB_Schema.json"
bank_json_path = "data/DUMMYDATA.json"

with open(ob_json_path, "r", encoding="utf-8") as f:
    ob_fields = json.load(f)

def build_ob_text(entry):
    schema = entry.get("schema", "")
    field = entry.get("field", "")
    description = entry.get("description", "")
    path = entry.get("path", "")
    method = entry.get("method", "")
    context = f"{method} {path}".strip()
    full = f"{context} {schema}.{field}: {description}".strip()
    return re.sub(r'\s+', ' ', full)

ob_texts = [build_ob_text(f) for f in ob_fields]

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

def preprocess_field(name):
    name = name.replace("[]", "")
    parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).replace("_", " ").lower().split(".")
    tokens = []
    for p in parts:
        for sub in p.split():
            tokens.append(abbreviation_map.get(sub, sub))
    return " ".join(tokens)

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
ob_embeddings = model.encode(ob_texts, convert_to_numpy=True)

embedding_dim = ob_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(ob_embeddings)

with open(bank_json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, list):
    bank_fields = raw
else:
    bank_fields = []
    for entry in raw.get("apiDataStrategySettings", []):
        mapping = entry.get("restJsonDynamicMapping", "[]")
        if isinstance(mapping, str):
            try:
                mapping = json.loads(mapping)
            except:
                continue
        for m in mapping:
            bank_fields.append(m.get("jsonPath", ""))

high_confidence_threshold = 0.85
medium_confidence_threshold = 0.70
min_confidence_gap = 0.1

mappings = []
mismatches = []
auto_mapped_count = 0

for name in bank_fields:
    cleaned = preprocess_field(name)
    query_embedding = model.encode([cleaned], convert_to_numpy=True)

    D, I = index.search(query_embedding, k=5)
    top_indices = I[0]
    top_scores = cosine_similarity(query_embedding, ob_embeddings[top_indices])[0]
    sorted_indices = np.argsort(top_scores)[::-1]

    best_idx = top_indices[sorted_indices[0]]
    top_score = top_scores[sorted_indices[0]]
    second_score = top_scores[sorted_indices[1]] if len(sorted_indices) > 1 else -1
    gap = top_score - second_score

    match = ob_fields[best_idx]

    if top_score >= high_confidence_threshold or (top_score >= medium_confidence_threshold and gap >= min_confidence_gap):
        mappings.append({
            "bank_field": name,
            "preprocessed": cleaned,
            "mapped_to": match.get("field", ""),
            "schema": match.get("schema", ""),
            "path": match.get("path", ""),
            "method": match.get("method", ""),
            "description": match.get("description", ""),
            "confidence": float(round(top_score, 3))
        })
        auto_mapped_count += 1
    else:
        suggestions = []
        for rank in sorted_indices[:3]:
            i = top_indices[rank]
            s = ob_fields[i]
            suggestions.append({
                "suggestion": s.get("field", "<missing>"),
                "schema": s.get("schema", ""),
                "path": s.get("path", ""),
                "method": s.get("method", ""),
                "description": s.get("description", ""),
                "confidence": float(round(top_scores[rank], 3))
            })
        mismatches.append({
            "bank_field": name,
            "preprocessed": cleaned,
            "top_suggestions": suggestions
        })

with open("field_mappings.json", "w", encoding="utf-8") as f:
    json.dump(mappings, f, indent=2, ensure_ascii=False)
with open("mismatches_log.json", "w", encoding="utf-8") as f:
    json.dump(mismatches, f, indent=2, ensure_ascii=False)

total = len(bank_fields)
print(f"Auto-mapped {auto_mapped_count}/{total} fields ({(auto_mapped_count / total) * 100:.2f}%)")
