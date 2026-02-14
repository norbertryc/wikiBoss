import typer
import logging
from config import config
from src.logging_config import setup_logging
from src.run.rag_pipeline import run_rag_pipeline


setup_logging(
    log_file=config.app_log_file,
    log_level=config.log_level,
    log_to_console=False
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def run(
        qdrant_collection_name: str =  config.get_collection_name("on_md_headers",
                                                                  suffix_override="e5_full")
):
    """"""

    run_rag_pipeline(
        qdrant_collection_name=qdrant_collection_name
    )

if __name__ == "__main__":
    app()