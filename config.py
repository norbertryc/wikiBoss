"""Global configuration."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories if missing
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# File paths
DUMP_FILE = "plwiki-latest-pages-articles.xml.bz2"
DUMP_PATH = RAW_DIR / DUMP_FILE
DUMP_URL = f"https://dumps.wikimedia.org/plwiki/latest/{DUMP_FILE}"
PARSED_JSONL = RAW_DIR / "parsed_articles.jsonl"
CLEANED_JSONL = PROCESSED_DIR / "cleaned_articles.jsonl"

# Processing settings
TOKENS_FOR_CHUNK = 512