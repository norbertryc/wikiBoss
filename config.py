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

# create directories if missing
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHUNKED_DIR.mkdir(parents=True, exist_ok=True)