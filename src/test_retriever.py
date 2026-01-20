import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .retriever import retriever
from config import COLLECTION_NAME

test_titles = [
    "Maria Skłodowska-Curie",
    "Stanisław Pilch",
    "Katowice",
    "Nereus",
    "Mikronezja",
    "Wisła",
    "Paracetamol",
    "Kraków",
    "Rower",
    "Mikołaj Kopernik",
]

for query_title in test_titles:
    print(f"\n=== RETRIEVER TEST FOR: {query_title} ===\n")

    results = retriever.retrieve(query_title)

    if not results:
        print("No results above threshold.")
        continue

    for i, hit in enumerate(results, 1):
        print(f"[{i}] SCORE: {hit.score:.4f} | Title: {hit.payload.get('title')}")

# =========================================================
# ONE-TIME: EXPORT ALL TITLES FROM COLLECTION TO JSON
# =========================================================

# data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# os.makedirs(data_dir, exist_ok=True)
# output_path = os.path.join(data_dir, "all_titles.json")

# all_docs, _ = retriever.client.scroll(
#     collection_name=COLLECTION_NAME,
#     limit=50000,
#     with_payload=True)

# all_titles = [doc.payload.get("title") for doc in all_docs]

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(all_titles, f, ensure_ascii=False, indent=2)

#     print(f"\nAll titles saved to {output_path}\n")