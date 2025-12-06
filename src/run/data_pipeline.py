from src.get_data import download_wikidump, parse_dump
from config import *
from src.logging_config import logger
from src.clean_data import Cleaner
from src.chunk_data import Chunker
from src.embed_data import QdrantIndexer


def data_pipeline(download: bool = True,
                  clean: bool = True,
                  chunk: bool = True,
                  embedding: bool = True,
                  num_lines: int = None,
                  chunking_strategy: str = "on_md_headers",
                  upload_batch: int = 20000):

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
        chunker = Chunker(input_path=CLEANED_JSONL,
                          output_path=CHUNKED_DIR / f"{chunking_strategy}_chunks.jsonl",
                          clear_output=True,
                          strategy=chunking_strategy,
                          load_on_init=True,
                          num_lines=num_lines)
        chunker.chunk()

    if embedding:
        indexer = QdrantIndexer(input_path=CHUNKED_DIR / f"{chunking_strategy}_chunks.jsonl",
                                load_on_init=True,
                                num_lines=num_lines)
        indexer.delete_collection(chunking_strategy)
        indexer.upload_points(chunking_strategy, upload_batch=upload_batch)

if __name__ == "__main__":
    data_pipeline()