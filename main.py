import uuid
import config
from tqdm import tqdm

from src.data_preparation import load_wiki_from_jsonl
from src.data_chunking_simple import Chunker
from src.vector_store import QdrantManager
from config import EMBEDDING_MODEL


if __name__ == "__main__":
    raw_data_full = load_wiki_from_jsonl(path=config.RAW_WIKI_JSONL)
    raw_data_limited = (article for i, article in enumerate(raw_data_full) if i < 50_000)

    chunker = Chunker(chunk_size=500, chunk_overlap=100)

    texts = []
    payloads = []
    ids = []   #UUID

    for article in tqdm(raw_data_limited, desc="Processing articles"):
        raw_chunks = chunker.chunk_text(article["text"])

        for c in raw_chunks:
            texts.append(c["chunk_text"])

            payloads.append({
                "article_id": article["id"],
                "title": article["title"],
                "url": article.get("url", ""),
                "chunk_index": c["chunk_index"],
            })

            ids.append(str(uuid.uuid4()))  # ⬅️ TO JEST UUID

    manager = QdrantManager()
    manager.delete_collection_if_exists("wiki_chunks")
    manager.create_collection_no_indexing("wiki_chunks", vector_size=384)

    embeddings = manager.generate_embeddings(texts, model_name=EMBEDDING_MODEL, batch_size=512)

    manager.upload_bulk_vectors(
        "wiki_chunks",
        vectors=embeddings,
        payloads=payloads,
        ids=ids   #UUID
    )

    manager.enable_indexing_after_upload("wiki_chunks")
