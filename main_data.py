import typer
import logging
from config import config
from src.logging_config import setup_logging
from src.run.data_pipeline import run_data_pipeline

setup_logging(
    log_file=config.app_log_file,
    log_level=config.log_level
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def run(
        download: bool = typer.Option(True, help="Download and parse Wikipedia dump"),
        clean: bool = typer.Option(True, help="Clean parsed articles"),
        chunk: bool = typer.Option(True, help="Split articles into chunks"),
        embedding: bool = typer.Option(True, help="Generate embeddings and upload to Qdrant"),
        num_lines: int = typer.Option(None, help="Max number of articles to process (None = all)"),
        start_from: int = typer.Option(0, help="Number of articles to skip from file start"),
        chunking_strategy: str = typer.Option("on_tokens",
                                              help="Chunking strategy: 'on_tokens' or 'on_md_headers'"),
        storage_suffix: str = typer.Option("", help="Suffix for chunk file and collection names"),
        delete_collection: bool = typer.Option(True, help="Delete existing Qdrant collection before upload"),
        clear_cleaned: bool = typer.Option(True, help="Delete existing cleaned file before processing"),
        clear_chunks: bool = typer.Option(True, help="Delete existing chunks file before processing"),
):
    """
    Run Wikipedia RAG data pipeline: download, clean, chunk, and embed articles.

    Examples:
        # Full pipeline with default 'on_tokens' chunking strategy
        python main_data.py

        # Get all data and parse to jsonl file, test further processing on 100 articles with md_headers strategy
        python main.py --num-lines 100 --chunking-strategy on_md_headers --storage-suffix _test

        # Only chunking step, all cleaned articles
        python main_data.py --no-download --no-clean --no-embedding

        # Process cleaned articles from line 10000, add to existing chunk file and qdrant collection
        python main_data.py \
        --no-download --no-clean \
        --start-from 10000 \
        --clear-chunks False \
        --delete-collection False \
        --chunking-strategy on_tokens
    """

    run_data_pipeline(
        download=download,
        clean=clean,
        chunk=chunk,
        embedding=embedding,
        data_num_lines_to_load=num_lines,
        start_loading_data_from=start_from,
        chunking_strategy=chunking_strategy,
        storage_suffix_override=storage_suffix,
        delete_collection_if_exists=delete_collection,
        clear_cleaner_output=clear_cleaned,
        clear_chunker_output=clear_chunks,
    )


if __name__ == "__main__":
    app()
