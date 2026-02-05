import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..src.retriever import retriever
from ..src.llm import ask_about
from config import COLLECTION_NAME

#one-time collect retirever results to notebook


test_titles = [
    "Maria Skłodowska-Curie",
    "Stanisław Pilch",
    "Katowice",
    "Nereus",
    "Mikronezja",
    "Wisła",
    "Paracetamol",
    "Kraków",
    "Rower górski",
    "Mikołaj Kopernik",
]

full_articles_titles = [
    "Maria Skłodowska-Curie",
    "Stanisław Pilch",
    "Katowice",
    "Paracetamol",
    "Rower górski",
]

llm_questions = [
    "Kim była Maria Skłodowska-Curie",
    "Co to jest Paracetamol",
    "Jakie są właściwości Paracetamolu",
    "Opisz Katowice",
    "Jakie są cechy roweru górskiego",
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
# ONE-TIME: COLLECT RETRIEVER & LLM RESULTS TO JSON
# =========================================================

# retriever results
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True) 
output_path = ARTIFACTS_DIR / "retriever_results.json"

all_results = {}

for query_title in test_titles:
    results = retriever.retrieve(query_title)

    all_results[query_title] = []

    for i, hit in enumerate(results[:5]):
        result = {
            "rank": i + 1,
            "score": float(hit.score),
            "title": hit.payload.get("title"),
            "text": hit.payload.get("text"),
        }
        all_results[query_title].append(result)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Retriever results saved to {output_path}")

# articles from vector store
full_articles_path = ARTIFACTS_DIR / "full_articles_results.json"
full_articles_results = {}

for title in full_articles_titles:
    docs = retriever.get_chunks_by_title(title) 
    full_text = "\n".join([d.payload.get("text", "") for d in docs])

    full_articles_results[title] = {
        "title": title,
        "full_text": full_text
    }

with open(full_articles_path, "w", encoding="utf-8") as f:
    json.dump(full_articles_results, f, ensure_ascii=False, indent=2)

print(f"Full articles saved to {full_articles_path}")

# llm answers
OUTPUT_DIR = ARTIFACTS_DIR 
os.makedirs(OUTPUT_DIR, exist_ok=True)

for query in llm_questions:
    answer = ask_about(query)
    file_name = "llm_answer_" + "".join(c if c.isalnum() else "_" for c in query) + ".json"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"question": query, "answer": answer}, f, ensure_ascii=False, indent=2)
    
    print(f"Saved answer to {file_path}\n")


# =========================================================
# ONE-TIME: EXPORT ALL TITLES FROM COLLECTION TO JSON
# =========================================================

# data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# os.makedirs(data_dir, exist_ok=True)
# output_path_all_titles = ARTIFACTS_DIR / "all_titles.json"

# all_docs, _ = retriever.client.scroll(
#     collection_name=COLLECTION_NAME,
#     limit=50000,
#     with_payload=True)

# all_titles = [doc.payload.get("title") for doc in all_docs]

# with open(output_path_all_titles, "w", encoding="utf-8") as f:
#     json.dump(all_titles, f, ensure_ascii=False, indent=2)

#     print(f"\nAll titles saved to {output_path}\n")