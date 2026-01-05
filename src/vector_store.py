from tqdm import tqdm
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from src.embedding_prep_bert import PolishBertCasedEmbedder



class QdrantManager:
    """
    Initialize Qdrant client.
    """
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        print(f"Connected with Qdrant on {url}")

    def create_collection_no_indexing(self, collection_name: str, vector_size: int = 768):
        """
        Create Qdrant collection with indexing disabled (fast upload mode).
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=0),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
            shard_number=2  # optional – parallel upload across shards
        )
        print(f"Created collection '{collection_name}' with indexing disabled.")


    def upload_bulk_vectors(
        self,
        collection_name: str,
        vectors,
        payloads=None,
        ids=None,
        parallel: int = 4
    ):
        """
        Uploads vectors to Qdrant in bulk mode.
        Assumes that the collection already exists.
        """
        print(f"Uploading {len(vectors)} vectors to '{collection_name}'...")

        for _ in tqdm(range(1), desc="Uploading vectors to Qdrant"):
            self.client.upload_collection(
                collection_name=collection_name,
                vectors=vectors,
                payload=payloads,
                ids=ids,
                parallel=parallel
            )

        print(f"Upload completed: {len(vectors)} vectors added to '{collection_name}'.")

    def enable_indexing_after_upload(self, collection_name: str):
        """
        Re-enable HNSW indexing and optimizers after upload.
        """
        self.client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(m=16),
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )
        print(f"Re-enabled indexing for '{collection_name}'")

    # def generate_embeddings(self, texts, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    #     """
    #     Generates sentence embeddings for a list of texts using SentenceTransformer.
    #     """
    
    #     model = SentenceTransformer(model_name)

    #     print(f"Generating embeddings for {len(texts)} texts...")
    #     embeddings = model.encode(texts, show_progress_bar=True)
    #     print(f"Generated embeddings with shape: {embeddings.shape}")

    #     return embeddings

    def generate_embeddings(self, texts, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", batch_size: int = 128):
        """
        Generates embeddings for a list of texts using SentenceTransformer in batches. NEW VERSION
        """
        model = SentenceTransformer(model_name)
        embeddings_all = []

        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            embeddings_all.extend(batch_embeddings)

        embeddings_all = np.array(embeddings_all)
        print(f"Generated embeddings with shape: {embeddings_all.shape}")

        return embeddings_all


    def generate_embeddings_bert(self, texts, batch_size=16):
        """
        Generates embeddings for a list of texts using the Polish BERT cased model cased.
        """
        embedder = PolishBertCasedEmbedder()
        embeddings = embedder.embed_batch(texts, batch_size=batch_size)
        return embeddings

    def delete_collection_if_exists(self, collection_name: str):
        """
        Deletes a Qdrant collection if it already exists.
        """
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing_collections:
            self.client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        else:
            print(f"Collection '{collection_name}' does not exist, skipping deletion.")



