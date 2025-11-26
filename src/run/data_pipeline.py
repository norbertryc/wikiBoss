from src.get_data import download_wikidump, parse_dump
from config import *
from src.logging import logger
from src.clean_data import Cleaner
from src.chunk_data import Chunker


def data_pipeline(download: bool = True,
                  clean: bool = True,
                  chunk: bool = True,
                  embedding: bool = True,
                  num_lines: int = None):

    if download:
        download_wikidump(DUMP_URL, DUMP_PATH)
        parse_dump(DUMP_PATH, PARSED_JSONL)

    if clean:
        cleaner = Cleaner(input_path=PARSED_JSONL,
                          output_path=CLEANED_JSONL,
                          clear_output=True,
                          load_on_init=True,
                          num_lines=num_lines)
        cleaner.clean()
        logger.info(cleaner.stats)

    if chunk:
        token_count_chunker = Chunker(input_path=CLEANED_JSONL,
                                      output_path=CHUNKED_DIR / "on_token_chunks.jsonl",
                                      clear_output=True,
                                      strategy="on_tokens",
                                      load_on_init=True,
                                      num_lines=num_lines)
        token_count_chunker.chunk()

        md_chunker = Chunker(input_path=CLEANED_JSONL,
                             output_path=CHUNKED_DIR / "md_chunks.jsonl",
                             clear_output=True,
                             strategy="on_md_headers",
                             load_on_init=True,
                             num_lines=num_lines)
        md_chunker.chunk()

if __name__ == "__main__":
    data_pipeline()