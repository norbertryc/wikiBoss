from datetime import datetime

# =========================
# WIKIPEDIA DATASET
# =========================
WIKIPEDIA_PL_DATASET = "chrisociepa/wikipedia-pl-20230401"

DATA_FOLDER = "data"
TODAY = datetime.now().strftime("%Y%m%d")

RAW_WIKI_JSON = f"{DATA_FOLDER}/raw_wiki_{TODAY}.json"
RAW_WIKI_JSONL = f"{DATA_FOLDER}/raw_wiki_pl.jsonl"

# =========================
# RETRIEVER / VECTOR STORE
# =========================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "wiki_chunks"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


