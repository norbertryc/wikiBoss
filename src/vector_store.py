from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer



def connect_qdrant():
    """
    Creates connection with Qdrant
    """
    client = QdrantClient(url="http://localhost:6333")
    print("✅ Connected with Qdrant on http://localhost:6333")
    return client

def create_wiki_collection(client, collection_name="wiki_chunks", vector_size=384):
    """
    Creates collection in Quadrant if does not exist yet.
    Vector size is an embedding size (384 for the model paraphrase-multilingual-MiniLM-L12-v2).
    """
    collections = client.get_collections().collections
    if collection_name not in [c.name for c in collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f" Created collection: {collection_name}")
    else:
        print(f"Collection '{collection_name}' already exists")


def upload_wiki_chunks(client, collection_name, chunks):
    """
    Converts text chunks into embeddings and uploads them to Qdrant.
    """
    print(" Generating embeddings and uploading to Qdrant...")

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    points = []
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk["chunk_text"]).tolist()
        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "title": chunk["title"],
                    "text": chunk["chunk_text"],
                    "url": chunk.get("url", "")
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f" Uploaded {len(points)} chunks to '{collection_name}'")