import config
from tqdm import tqdm

from src.data_preparation import ( 
    load_wiki_from_jsonl
    )

from src.data_chunking_simple import simple_chunk_wikipedia_article
from src.vector_store import QdrantManager


if __name__ == "__main__":
    raw_data_full = load_wiki_from_jsonl(path=config.RAW_WIKI_JSONL)
    raw_data_limited = (article for i, article in enumerate(raw_data_full) if i < 50_000)


    simple_chunking = []
    for article in tqdm(raw_data_limited, desc="Processing articles"):
        article_chunks = simple_chunk_wikipedia_article(article=article,
                                                        chunk_size=500,
                                                        chunk_overlap=100)
        simple_chunking.extend(article_chunks)

    manager = QdrantManager()
    manager.delete_collection_if_exists("wiki_chunks")
    manager.create_collection_no_indexing("wiki_chunks", vector_size=384)

    payloads = [{"text": chunk["chunk_text"], "title": chunk["title"]} for chunk in simple_chunking]
    texts = [p["text"] for p in payloads]

    embeddings = manager.generate_embeddings(texts, batch_size=512)

    manager.upload_bulk_vectors("wiki_chunks", vectors=embeddings, payloads=payloads)

    manager.enable_indexing_after_upload("wiki_chunks")
