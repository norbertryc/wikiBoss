# === TEST: sprawdź pierwsze chunki dla wszystkich osób z configu ===
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.retriever import Retriever
from src.vector_store import QdrantManager
from config import Config

cfg = Config()

qdrant_manager = QdrantManager(url=cfg.qdrant_url)

retriever = Retriever(
    qdrant_manager=qdrant_manager,
    collection_name=cfg.base_collection_name,
    embedding_model=cfg.embedding_model,
    top_k=50 
)

print("\n=== TEST LEADÓW DLA WSZYSTKICH OSÓB ===")

for title in cfg.target_titles:
    print(f"\n\n===== {title} =====")
    docs = retriever.get_chunks_by_title(title)

    if not docs:
        print("Brak chunków w Qdrant!")
        continue

    docs_sorted = sorted(docs, key=lambda d: d.payload["chunk_index"])

    for i, d in enumerate(docs_sorted[:3], start=1):
        print(f"\n--- CHUNK {i} ---")
        print(d.payload["text"][:300])
