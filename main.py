import os
from src.get_data import download_wikidump, parse_dump
from src.clean_and_chunk import process_data
from src.config import *


def main():
    download_wikidump(DUMP_URL, DUMP_PATH)
    parse_dump(DUMP_PATH, JSON_RAW)
    process_data(JSON_RAW, os.path.join(PROCESSED_DIR, "chunks_by_section.jsonl"))


if __name__ == "__main__":
    main()