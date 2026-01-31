"""Global configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKED_DIR = PROCESSED_DIR / "chunked"

# file_paths
DUMP_FILE = "plwiki-latest-pages-articles.xml.bz2"
DUMP_PATH = RAW_DIR / DUMP_FILE
DUMP_URL = f"https://dumps.wikimedia.org/plwiki/latest/{DUMP_FILE}"
PARSED_JSONL = RAW_DIR / "parsed_articles.jsonl"
CLEANED_JSONL = PROCESSED_DIR / "cleaned_articles.jsonl"
LOG_FILE = PROJECT_ROOT / "logs.txt"

# models
# HUGGING_FACE_MODEL = {"model": "dkleczek/bert-base-polish-cased-v1", "is_sentence_transformers_model": False}
# HUGGING_FACE_MODEL = {"model": "sdadas/st-polish-paraphrase-from-mpnet", "is_sentence_transformers_model": True}
HUGGING_FACE_MODEL = {"model": "intfloat/multilingual-e5-base", "is_sentence_transformers_model": True}

# Qdrant
QDRANT_CONFIG = {"host": "localhost", "port": 6333}
HNSW_CONFIG = {"m": 32, "ef_construct": 200} # 2-5M chunks, adjust to the collection and hardware
