import typer
from src.run.rag_pipeline import rag_pipeline


app = typer.Typer()


@app.command()
def run(
        qdrant_collection_name: str = "on_md_headers_e5_full",
        device: str = "cuda",
        half_precision: bool = False
):
    """"""

    rag_pipeline(
        qdrant_collection_name=qdrant_collection_name,
        device=device,
        half_precision=half_precision
    )

if __name__ == "__main__":
    app()