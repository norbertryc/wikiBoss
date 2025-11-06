import numpy as np
import time
from tqdm import tqdm
from qdrant_client import QdrantClient, models


def connect_qdrant():
    """
    Creates connection with Qdrant
    """
    client = QdrantClient(url="http://localhost:6333")
    print("Connected with Qdrant on http://localhost:6333")
    return client


def create_collection_no_indexing(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """
    Create Qdrant collection with indexing disabled (fast upload mode).
    """
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        shard_number=2  # optional – parallel upload across shards
    )
    print(f"Created collection '{collection_name}' with indexing disabled.")


def upload_bulk_vectors(
    client: QdrantClient,
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
        client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payloads,
            ids=ids,
            parallel=parallel
        )

    print(f"Upload completed: {len(vectors)} vectors added to '{collection_name}'.")


def enable_indexing_after_upload(client: QdrantClient, collection_name: str):
    """
    Re-enable HNSW indexing and optimizers after upload.
    """
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(m=16),
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    )
    print(f"Re-enabled indexing for '{collection_name}'")

def generate_embeddings(texts, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Generates sentence embeddings for a list of texts using SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    return embeddings

def delete_collection_if_exists(client, collection_name: str):
    """
    Deletes a Qdrant collection if it already exists.
    """
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    else:
        print(f"Collection '{collection_name}' does not exist, skipping deletion.")



