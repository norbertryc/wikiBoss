import os

DUMP_FILE = "plwiki-latest-pages-articles.xml.bz2"
DUMP_PATH = os.path.join("data", "raw", DUMP_FILE)
DUMP_URL = f"https://dumps.wikimedia.org/plwiki/latest/{DUMP_FILE}"
JSON_RAW = os.path.join("data", "raw", "plwiki_latest_articles.jsonl")