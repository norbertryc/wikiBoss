import config

from src.data_preparation import download_wiki_data, save_wiki_to_json, save_wiki_to_csv
from src.data_chunking_advanced import chunk_wikipedia_article
from src.data_chunking_simple import simple_chunk_wikipedia_article
from src.vector_store import connect_qdrant, create_wiki_collection, upload_wiki_chunks


#Load raw wikipedia data
raw_data = download_wiki_data(dataset_name=config.WIKIPEDIA_PL_DATASET, limit=40)

#Simple chunking:
simple_chunking = []
for article in raw_data:
    article_chunks = simple_chunk_wikipedia_article(article=article,
                                                    chunk_size=500,
                                                    chunk_overlap=100)
    simple_chunking.extend(article_chunks)

#Chunking advanced:
# advanced_chunking = []
# for article in raw_data:
#     article_chunks = chunk_wikipedia_article(article=article,
#                                     chunk_size=600,
#                                     chunk_overlap=100)
#     advanced_chunking.extend(article_chunks) 


# for article in raw_data:
#     chunks = chunk_wikipedia_article(article, chunk_size=600, chunk_overlap=100)
#     print(f"{article['title']}: {len(chunks)} chunk(s)")

# for i, chunk in enumerate(simple_chunking[:40]):  # pokaże pierwsze 40 chunków
#     print(f"\n--- Chunk {i+1} (Article: {chunk['title']}) ---")
#     print(f"URL: {chunk.get('url', 'Brak linku')}")
#     print(chunk['chunk_text'][:500])

client = connect_qdrant()
create_wiki_collection(client, collection_name="wiki_chunks", vector_size=384)
upload_wiki_chunks(client, "wiki_chunks", simple_chunking)

# --- Print data in Qdrant ---
print("\n See data in Qdrant:")
scroll_results = client.scroll(
    collection_name="wiki_chunks",
    limit=3  # take 3 chunks to display
)

for point in scroll_results[0]:
    payload = point.payload
    print(f" Title: {payload.get('title')}")
    print(f" URL: {payload.get('url')}")
    text = payload.get("chunk_text") or payload.get("text") or ""
    print(f" Text: {text[:250]}...")

if __name__ == "__main__":
    pass