import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K


class SimpleRetriever:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_query(self, query: str):
        return self.model.encode(query)

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_vector = self.embed_query(query)

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

        return results


if __name__ == "__main__":
    retriever = SimpleRetriever()

    query = "Kim był Mikolaj Kopernik?"
    results = retriever.retrieve(query)

    print("\n=== WYNIKI ===\n")
    for i, hit in enumerate(results, 1):
        print(f"[{i}] SCORE: {hit.score:.4f}")
        print(f"Tytuł: {hit.payload.get('title')}")
        print(f"Tekst:\n{hit.payload.get('text')[:500]}")
        print("-" * 80)

