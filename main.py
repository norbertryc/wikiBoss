import config

from src.data_preparation import ( 
    get_wikipedia_article_count,
    load_wiki_from_json
    )

from src.data_chunking_simple import simple_chunk_wikipedia_article
from src.vector_store import QdrantManager

if __name__ == "__main__":
    raw_data_full = load_wiki_from_json(path=config.RAW_WIKI_JSON)

    total_articles = get_wikipedia_article_count(dataset_name=config.WIKIPEDIA_PL_DATASET)

    simple_chunking = []
    for article in raw_data_full:
        article_chunks = simple_chunk_wikipedia_article(article=article,
                                                        chunk_size=500,
                                                        chunk_overlap=100)
        simple_chunking.extend(article_chunks)

    manager = QdrantManager()
    manager.delete_collection_if_exists("wiki_chunks")
    manager.create_collection_no_indexing("wiki_chunks", vector_size=384)

    payloads = [{"text": article["text"], "title": article["title"]} for article in raw_data_full]
    texts = [p["text"] for p in payloads]
    # embeddings = manager.generate_embeddings(texts)
    embeddings = manager.generate_embeddings_bert(texts, batch_size=16)

    manager.upload_bulk_vectors("wiki_chunks", vectors=embeddings, payloads=payloads)

    manager.enable_indexing_after_upload("wiki_chunks")
