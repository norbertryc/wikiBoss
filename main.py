from src.get_data import download_wikidump, parse_dump
from config import *
from src.logging import logger
from src.clean_data import Cleaner
from src.chunk_data import Chunker


def main():
    download_wikidump(DUMP_URL, DUMP_PATH)
    parse_dump(DUMP_PATH, PARSED_JSONL)

    # cleaner = Cleaner(input_path=PARSED_JSONL,
    #                   output_path=CLEANED_JSONL,
    #                   clear_output=True)
    # cleaner.load(num_lines=5000)
    # cleaner.load()
    # cleaner.clean()
    # logger.info(cleaner.stats)

    token_count_chunker = Chunker(input_path=CLEANED_JSONL,
                                  output_path=CHUNKED_DIR / "on_token_chunks.jsonl",
                                  clear_output=True,
                                  strategy="on_tokens")
    token_count_chunker.load(num_lines=None)
    token_count_chunker.chunk()

    md_chunker = Chunker(input_path=CLEANED_JSONL,
                         output_path=CHUNKED_DIR / "md_chunks.jsonl",
                         clear_output=True,
                         strategy="on_md_headers")
    md_chunker.load(num_lines=None)
    md_chunker.chunk()

if __name__ == "__main__":
    main()