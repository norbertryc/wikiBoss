import uuid
from tqdm import tqdm 

from config import Config
from src.data_download import load_wiki_from_jsonl
from src.data_chunking_simple import Chunker
from src.vector_store import QdrantManager


def main():
    cfg = Config()

    raw_data_full = load_wiki_from_jsonl(path=cfg.raw_wiki_jsonl)

    raw_data_limited = (
        article for i, article in enumerate(raw_data_full) if i < 100
    )

    chunker = Chunker(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap
    )

    texts = []
    payloads = []
    ids = []

    for article in tqdm(raw_data_limited, desc="Processing articles", unit="art"):
        article_id = article.get("id", str(uuid.uuid4()))
    
        for c in tqdm(chunker.chunk_text(article["text"]),
                      desc="Chunking",
                      leave=False,
                      unit="chunk"):
            texts.append(c["chunk_text"])

            payloads.append({
                "article_id": article_id,
                "title": article["title"],
                "url": article.get("url", ""),
                "chunk_index": c["chunk_index"],
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "tokenization_variant": cfg.tokenization_variant,
                "text": c["chunk_text"],
            })
            ids.append(str(uuid.uuid4()))
            

    manager = QdrantManager()
    collection_name = "wiki_test_after_code_review"

    # manager.delete_collection_if_exists(collection_name=collection_name)
    
    manager.create_collection_no_indexing(
        collection_name=collection_name,
        vector_size=cfg.vector_size
    )

    manager.upload_collection(
        collection_name=collection_name,
        texts=texts,
        payloads=payloads,
        ids=ids,
        model_name=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
    )


if __name__ == "__main__":
    main()

