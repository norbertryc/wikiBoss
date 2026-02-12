import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.retriever import Retriever
from app_assistant import Assistant
from config import Config

# =========================================================
# CONFIG
# =========================================================

cfg = Config()


retriever = Retriever(
    qdrant_url=cfg.qdrant_url,
    collection_name=cfg.base_collection_name,
    embedding_model=cfg.embedding_model,
    top_k=cfg.top_k
)

assistant = Assistant(cfg) 

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
    "Opisz Warszawę",  
]

ARTIFACTS_DIR = Path("artifacts")

# =========================================================
# TEST RETRIEVER and CHUNKS PRINTING
# =========================================================

for title in test_titles + ["Warszawa"]: 
    print(f"\n=== RETRIEVER TEST FOR: {title} ===")
    docs = retriever.get_chunks_by_title(title)
    print(f"Liczba chunków: {len(docs)}")

    if not docs:
        print("Brak chunków w vector store.")
        continue

    for i, d in enumerate(docs[:5], 1):  # pokaż pierwsze 5 chunków
        text_snippet = d.payload.get("text", "")[:150]
        print(f"[{i}] {text_snippet}...")

# =========================================================
# RETRIEVER RESULTS TO JSON
# =========================================================
print(f"\n=== RETRIEVER TEST TO JSON ===")

retriever_results_path = ARTIFACTS_DIR / "retriever_results.json"
all_results = {}

for title in test_titles + ["Warszawa"]:
    docs = retriever.get_chunks_by_title(title)
    all_results[title] = []

    for i, d in enumerate(docs[:5]):
        all_results[title].append({
            "rank": i + 1,
            "title": d.payload.get("title"),
            "score": float(d.score) if hasattr(d, "score") else None,
            "text": d.payload.get("text", ""),
        })

with open(retriever_results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\nRetriever results saved to {retriever_results_path}")

# =========================================================
# FULL ARTICLES
# =========================================================

# full_articles_path = ARTIFACTS_DIR / "full_articles_results.json"
# full_articles_results = {}

# for title in full_articles_titles + ["Warszawa"]: 
#     docs = retriever.get_chunks_by_title(title)
#     full_text = "\n".join(d.payload.get("text", "") for d in docs)

#     full_articles_results[title] = {
#         "title": title,
#         "full_text": full_text,
#         "num_chunks": len(docs)
#     }

# with open(full_articles_path, "w", encoding="utf-8") as f:
#     json.dump(full_articles_results, f, ensure_ascii=False, indent=2)

# print(f"Full articles saved to {full_articles_path}")

# # =========================================================
# #  LLM ANSWERS
# # =========================================================

# print("\n=== LLM ANSWERS ===")
# for query in llm_questions:
#     answer = assistant.generate_answer(query)
#     file_name = "llm_answer_" + "".join(c if c.isalnum() else "_" for c in query) + ".json"
#     file_path = ARTIFACTS_DIR / file_name

#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump({"question": query, "answer": answer}, f, ensure_ascii=False, indent=2)

#     print(f"\nQuestion: {query}\nAnswer: {answer}\nSaved to {file_path}")

