from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

class Document:
    """Minimal Document class to hold page content and metadata for LLM retrieval"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class Retriever:
    """Retriever class to fetch relevant documents for LLM"""

    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.top_k = TOP_K

    def embed_query(self, query: str):
        """Convert query to embedding vector"""
        return self.model.encode(query)

    def retrieve(self, query: str, top_k=None):
        """Return top-k search results from Qdrant"""
        top_k = top_k or self.top_k
        query_vector = self.embed_query(query)
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.65
        )
        return results

    def get_relevant_documents(self, query: str, top_k=None):
        """Return documents in format expected by LLM"""
        results = self.retrieve(query, top_k=top_k)
        documents = []
        for hit in results:
            content = hit.payload.get("content", "")
            metadata = {"title": hit.payload.get("title", ""),
                        "score": hit.score}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

retriever = Retriever()
