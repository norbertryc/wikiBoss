from qdrant_client import QdrantClient
from collections import Counter

client = QdrantClient(url="http://localhost:6333")

# collection_1 = "wiki_chunks"
# collection_2 = "wiki_chunks_cs500_co100_toksimple_embparaphrase-multilingual-MiniLM-L12-v2"  

# points = client.scroll(collection_name=collection_2, limit=10, with_vectors=True)


# for i, p in enumerate(points):
#     print(f"--- POINT #{i} type={type(p)} ---")
#     print(repr(p))
#     if i >= 1:
        # break

collections = [
    "wiki_chunks",
    "wiki_chunks_cs500_co100_toksimple_embparaphrase-multilingual-MiniLM-L12-v2",
    "wiki_test_100"
]

# for coll in collections:
#     print(f"\n=== {coll} ===")
#     pts = client.scroll(collection_name=coll, limit=10, with_vectors=False)
#     for p in pts:
#         rec = p[0] if isinstance(p, (list, tuple)) else p
#         payload = getattr(rec, "payload", None) or (rec.get("payload") if isinstance(rec, dict) else {})
#         print( f"article_id: {payload.get('article_id')}\n" 
#               f"title: {payload.get('title')}\n" 
#               f"url: {payload.get('url')}\n" 
#               f"chunk_index: {payload.get('chunk_index')}\n" 
#               f"chunk_size: {payload.get('chunk_size')}\n" 
#               f"chunk_overlap: {payload.get('chunk_overlap')}\n"
#               f"tokenization_variant: {payload.get('tokenization_variant')}\n" 
#               f"text: {payload.get('text')}" 
#               )
    

for coll in collections:
    has = 0
    no = 0
    pts_pages = client.scroll(collection_name=coll, limit=500, with_vectors=False)  # zwraca strony (listy)
    for page in pts_pages:                      # każda strona to lista rekordów
        for p in page:                          # iteruj po rekordach w stronie
            rec = p[0] if isinstance(p, (list, tuple)) else p
            payload = getattr(rec, "payload", None) or (rec.get("payload") if isinstance(rec, dict) else {})
            if payload.get("text") is not None:
                has += 1
            else:
                no += 1
    total = has + no
    print(f"\n=== {coll} ===")
    print(f"checked: {total}")
    print(f"with text: {has}")
    print(f"without text: {no}")


# coll = "wiki_test_100"

# counter = Counter()
# pages = client.scroll(collection_name=coll, limit=500, with_vectors=False)
# for page in pages:
#     for item in page:
#         rec = item[0] if isinstance(item, (list, tuple)) else item
#         payload = getattr(rec, "payload", None) or (rec.get("payload") if isinstance(rec, dict) else {})
#         for k in payload.keys():
#             counter[k] += 1

# print("Top payload keys and counts:")
# for k, v in counter.most_common():
#     print(k, v)

