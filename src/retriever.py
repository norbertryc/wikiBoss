from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer


class Retriever:
    """Retriever class responsible only for vector search in Qdrant"""

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model: str,
        top_k: int = 5,
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        self.top_k = top_k

    def embed_query(self, query: str) -> list[float]:
        """Convert query to embedding vector"""
        return self.model.encode(query).tolist()

    def retrieve(self, query: str, top_k: int | None = None):
        """Return raw Qdrant search results (ScoredPoint)"""
        limit = top_k or self.top_k

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=self.embed_query(query),
            limit=limit,
        )

    def get_chunks_by_title(self, title: str):
        """Return all chunks belonging to a specific Wikipedia article"""
        title_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="title",
                    match=qdrant_models.MatchValue(value=title),
                )
            ]
        )

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=50000,
            scroll_filter=title_filter,
        )

        return results

