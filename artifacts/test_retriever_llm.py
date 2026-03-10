import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.retriever import Retriever
from src.vector_store import QdrantManager
from app_assistant import Assistant
from config import Config

# =========================================================
# CONFIG
# =========================================================

cfg = Config()

qdrant_manager = QdrantManager(url=cfg.qdrant_url)

retriever = Retriever(
    qdrant_manager=qdrant_manager,
    collection_name=cfg.base_collection_name,
    embedding_model=cfg.embedding_model,
    top_k=cfg.top_k
)

assistant = Assistant(cfg) 

# =========================================================
# TEST: SPRAWDŹ, CZY W QDRANT SĄ DATY ŚMIERCI
# =========================================================

test_queries = [
    "zm. 1543",
    "zm. Mikołaj Kopernik",
    "Data śmierci (zm.) Mikołaj Kopernik",
    "Mikołaj Kopernik zm.",
]

print("\n=== TEST RETRIEVERA: DATY ŚMIERCI ===")
for q in test_queries:
    results = retriever.retrieve(q)
    print(f"\nZapytanie: {q}")
    print("Liczba trafień:", len(results))
    if results:
        print("Fragment:", results[0].payload.get("text", "")[:200])


test_titles = [ 
    "Adam Mickiewicz", 
    "Stefan Banach", 
    "Maria Skłodowska-Curie", 
    "Mikołaj Kopernik", 
    "Cyprian Kamil Norwid", ]


llm_questions = [   
    "Kto był starszy w momencie śmierci: Hugo Steinhaus czy Stanisław Ulam? O ile lat?",
    "Kto żył dłużej: Józef Ignacy Kraszewski czy Stefan Banach? O ile lat?",
    "Kto zmarł wcześniej: Mikołaj Kopernik czy Jan Heweliusz?",
     ]


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# # =========================================================
# # TEST RETRIEVER AND CHUNKS PRINTING
# # =========================================================

# for title in test_titles: 
#     print(f"\n=== RETRIEVER TEST FOR: {title} ===")
#     docs = retriever.get_chunks_by_title(title)
#     print(f"Liczba chunków: {len(docs)}")

#     if not docs:
#         print("Brak chunków w vector store.")
#         continue

#     for i, d in enumerate(docs[:5], 1):  # pokaż pierwsze 5 chunków
#         text_snippet = d.payload.get("text", "")[:150]
#         print(f"[{i}] {text_snippet}...")

# # =========================================================
# # RETRIEVER RESULTS TO JSON
# # =========================================================
# print(f"\n=== RETRIEVER TEST TO JSON ===")

# retriever_results_path = ARTIFACTS_DIR / "retriever_results_romantics_and_scientists.json"
# all_results = {}

# for title in test_titles:
#     docs = retriever.get_chunks_by_title(title)
#     all_results[title] = []

#     for i, d in enumerate(docs[:5]):
#         all_results[title].append({
#             "rank": i + 1,
#             "title": d.payload.get("title"),
#             "score": float(d.score) if hasattr(d, "score") else None,
#             "text": d.payload.get("text", ""),
#         })

# with open(retriever_results_path, "w", encoding="utf-8") as f:
#     json.dump(all_results, f, ensure_ascii=False, indent=2)

# print(f"\nRetriever results saved to {retriever_results_path}")

# # =========================================================
# #  LLM ANSWERS
# # =========================================================

print("\n=== LLM ANSWERS ===")
for query in llm_questions:
    answer = assistant.generate_answer(query)
    file_name = "llm_answer_" + "".join(c if c.isalnum() else "_" for c in query) + ".json"
    file_path = ARTIFACTS_DIR / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"question": query, "answer": answer}, f, ensure_ascii=False, indent=2)

    print(f"\nQuestion: {query}\nAnswer: {answer}\nSaved to {file_path}")

