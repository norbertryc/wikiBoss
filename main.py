from src.get_data import download_wikidump, parse_dump
from config import *
from src.logging import logger
from src.clean_data import Cleaner


def main():
    download_wikidump(DUMP_URL, DUMP_PATH)
    parse_dump(DUMP_PATH, PARSED_JSONL)

    cleaner = Cleaner(input_path=PARSED_JSONL,
                      output_path=CLEANED_JSONL,
                      clear_output=True)
    # cleaner.load(num_lines=5000)
    cleaner.load()
    cleaner.clean()
    logger.info(cleaner.stats)

if __name__ == "__main__":
    main()