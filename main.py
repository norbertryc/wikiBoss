from src.get_data import download_wikidump, parse_dump
from config import *


def main():
    download_wikidump(DUMP_URL, DUMP_PATH)
    parse_dump(DUMP_PATH, PARSED_JSONL)

if __name__ == "__main__":
    main()