import typer
from src.run.data_pipeline import data_pipeline

app = typer.Typer()

@app.command()
def run(num_lines: int = typer.Option(None),
        download: bool = typer.Option(True),
        clean: bool = typer.Option(True),
        chunk: bool = typer.Option(True),
        embedding: bool = typer.Option(True),
        chunking_strategy: str = typer.Option("on_md_headers"),
        upload_batch: int = typer.Option(20000),
        ):

    data_pipeline(
        download=download,
        clean=clean,
        chunk=chunk,
        num_lines=num_lines,
        embedding=embedding,
        chunking_strategy=chunking_strategy,
        upload_batch=upload_batch
    )

if __name__ == "__main__":
    app()