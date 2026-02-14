import logging
import typer
from config import config, EmbeddingModelConfig
from src.vector_store import QdrantManager
from src.embedding_engine import EmbeddingEngine
from src.assistant import WikiBoss

logger = logging.getLogger(__name__)


def run_rag_pipeline(
        embedding_model: EmbeddingModelConfig = None,
        qdrant_collection_name: str = None,
):
    """Interactive RAG pipeline."""

    # use defaults if not provided
    if embedding_model is None:
        embedding_model = config.embedding_model
    if qdrant_collection_name is None:
        qdrant_collection_name = config.get_collection_name(chunking_method="on_md_headers",
                                                            suffix_override="e5_full")

    logger.info(f"Starting RAG pipeline with collection: {qdrant_collection_name}")

    # validate Groq API key
    groq_api_key = config.require_groq_key()

    # initialize components
    embedding_engine = EmbeddingEngine(
        model_config=embedding_model,
        device=config.device,
        use_fp16=config.half_precision
    )

    qdrant = QdrantManager(
        embedding_engine=embedding_engine,
        collection_name=qdrant_collection_name
    )

    assistant = WikiBoss(
        qdrant_manager=qdrant,
        api_key=groq_api_key,
        model=config.primary_model,
        fallback_model=config.fallback_model
    )


    typer.echo("Zadaj pytanie (wpisz 'quit' aby wyjść)")

    while True:
        try:
            question = typer.prompt("Twoje pytanie").strip()

            if question.lower() == "quit":
                typer.echo("Do widzenia!")
                break

            if not question:
                continue

            answer = assistant.invoke(question)
            typer.echo(f"\nWikiBoss: {answer}\n")

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            typer.echo(f"Error: {e}\n", err=True)


if __name__ == "__main__":
    run_rag_pipeline()