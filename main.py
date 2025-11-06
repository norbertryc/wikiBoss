import config

from src.data_preparation import (
    download_wiki_data,  
    get_wikipedia_article_count
    )

from src.data_chunking_simple import simple_chunk_wikipedia_article
from src.vector_store import (
    connect_qdrant,
    create_collection_no_indexing,
    generate_embeddings,
    upload_bulk_vectors,
    delete_collection_if_exists,
    enable_indexing_after_upload
)

if __name__ == "__main__":
    raw_data_full = download_wiki_data(dataset_name=config.WIKIPEDIA_PL_DATASET, limit=3000)

    total_articles = get_wikipedia_article_count(dataset_name=config.WIKIPEDIA_PL_DATASET)

    simple_chunking = []
    for article in raw_data_full:
        article_chunks = simple_chunk_wikipedia_article(article=article,
                                                        chunk_size=500,
                                                        chunk_overlap=100)
        simple_chunking.extend(article_chunks)

    client = connect_qdrant()
    delete_collection_if_exists(client, "wiki_chunks")
    create_collection_no_indexing(client, collection_name="wiki_chunks", vector_size=384)

    sample_payloads = [{"text": article["text"], "title": article["title"]} for article in raw_data_full]
    sample_embeddings = generate_embeddings(sample_payloads)

    upload_bulk_vectors(
        client,
        collection_name="wiki_chunks",
        vectors=sample_embeddings,
        payloads=sample_payloads 
    )

    enable_indexing_after_upload(client, "wiki_chunks")
