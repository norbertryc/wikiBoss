import logging
from config import config
from src.get_data import download_wikidump, parse_dump
from src.clean_data import Cleaner
from src.chunk_data import Chunker
from src.vector_store import QdrantManager
from src.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)

def run_data_pipeline(
        # Workflow control
        download: bool = True,
        clean: bool = True,
        chunk: bool = True,
        embedding: bool = True,

        # Data loading options
        data_num_lines_to_load: int = None,
        start_loading_data_from: int = 0,

        # Chunking, storage name
        chunking_strategy: str = "on_tokens",
        storage_suffix_override: str|None = None,  # Override auto-suffix

        # File handling
        delete_collection_if_exists: bool = True,
        clear_cleaner_output: bool = True,
        clear_chunker_output: bool = True,
                      ):
    """
    End-to-end data pipeline for Wikipedia RAG system: download, clean, chunk, and embed.

    Pipeline steps (all optional):
    1. Download latest Wikipedia dump and parse to JSONL
    2. Clean and convert MediaWiki to Markdown
    3. Split articles into chunks using specified strategy
    4. Generate embeddings and upload to Qdrant vector database

    Args:
        download (bool): If True, downloads and parses Wikipedia dump (default: True).
                        If dump/parsed files already exist, these steps are skipped automatically
                        with comments only logged.
        clean (bool): If True, cleans parsed articles (default: True)
        chunk (bool): If True, splits cleaned articles into chunks (default: True)
        embedding (bool): If True, embeds chunks and uploads to Qdrant (default: True)
        data_num_lines_to_load (int, optional): Max number of articles to load from input file.
                                               If None, processes entire file. Useful for testing
                                               on small article samples. Note: final record count
                                               may be lower due to processing filters (check logs).
        start_loading_data_from (int): Number of articles to skip from file start (default: 0).
                                      Useful for testing or rerun on fail.
        chunking_strategy (str): Chunking method - 'on_tokens' or 'on_md_headers' (default: 'on_tokens').
                                Must be consistent between chunk and embedding steps.
        storage_suffix_override (str): Suffix appended to chunk file and collection names. If not provided,
                                    embedding model name is used.
        delete_collection_if_exists (bool): If True, deletes existing Qdrant collection
                                           before upload (default: True)
        clear_cleaner_output (bool): If True, deletes existing cleaned file before processing (default: True).
                                    If False, new records are appended to existing file.
        clear_chunker_output (bool): If True, deletes existing chunks file before processing (default: True).
                                    If False, new chunks are appended to existing file.

    Note:
        - Download step: If dump or parsed files exist, download/parse is skipped automatically
          (logged only). This prevents re-downloading on subsequent runs.

        - Clean/Chunk steps: By default (clear_*_output=True), existing output files are deleted
          and processing runs from scratch. Set to False to append to existing files.

        - Testing: Use data_num_lines_to_load and/or start_loading_data_from to test pipeline on
          small article samples. The final record count may be lower than requested due to
          invalid records being filtered during processing (automatically handled, check logs).

        - Naming consistency: chunking_strategy and storage_name_suffix determine both chunks
          filename and Qdrant collection name. When running chunk and embedding separately,
          ensure these parameters match.

          Example filenames/collections:
          - strategy='on_tokens', suffix=None → 'on_tokens_chunks_model_name.jsonl' / collection 'on_tokens_model_name'
          - strategy='on_md_headers', suffix='_test' → 'on_md_headers_chunks_test.jsonl' / collection 'on_md_headers_test'
    """
    # create processed data directories if missing
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.chunked_dir.mkdir(parents=True, exist_ok=True)

    # download latest wikidump and parse to jsonl file
    if download:
        download_wikidump(config.dump_url, config.dump_path)
        parse_dump(config.dump_path, config.parsed_jsonl)

    # clean parsed mediawiki articles
    if clean:
        cleaner = Cleaner(input_path=config.parsed_jsonl,
                          output_path=config.cleaned_jsonl,
                          clear_output=clear_cleaner_output,
                          num_lines=data_num_lines_to_load)
        cleaner.clean()

    # split data into chunks using a selected strategy, embed and store them in a vector database
    if chunk or embedding:
        embedding_engine = EmbeddingEngine(config.embedding_model,
                                           device=config.device,
                                           use_fp16=config.half_precision)

        chunks_file = config.get_chunked_jsonl_path(chunking_method=chunking_strategy,
                                                    suffix_override=storage_suffix_override)

        if chunk:
            chunker = Chunker(input_path=config.cleaned_jsonl,
                              output_path=chunks_file,
                              embedding_engine=embedding_engine,
                              clear_output=clear_chunker_output,
                              strategy=chunking_strategy,
                              num_lines=data_num_lines_to_load,
                              start_loading=start_loading_data_from)
            chunker.chunk()

        if embedding:
            qdrant = QdrantManager(input_path=chunks_file,
                                   num_lines=data_num_lines_to_load,
                                   start_loading=start_loading_data_from,
                                   embedding_engine=embedding_engine,
                                   encoding_batch_size=config.encoding_batch_size)

            collection_name = config.get_collection_name(chunking_method=chunking_strategy,
                                                         suffix_override=storage_suffix_override)
            if delete_collection_if_exists:
                qdrant.delete_collections(collection_name)
            qdrant.upload_points(collection_name,
                                 upload_batch_size=config.upload_batch_size)

    logger.info("END")

if __name__ == "__main__":
    run_data_pipeline()