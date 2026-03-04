from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    # =========================
    # DATA
    # =========================

    raw_wiki_jsonl: Path = Path("data/raw_wiki_polish_romantics_and_scientists.jsonl")

    target_titles: list[str] = [
    "Adam Mickiewicz",
    "Juliusz Słowacki",
    "Cyprian Kamil Norwid",
    "Zygmunt Krasiński",
    "Aleksander Fredro",
    "Józef Ignacy Kraszewski",
    "Maria Skłodowska-Curie",
    "Mikołaj Kopernik",
    "Jan Heweliusz",
    "Stefan Banach",
    "Hugo Steinhaus",
    "Stanisław Ulam",
]

    # =========================
    # CHUNKING
    # =========================
    chunk_size: int = 1000
    chunk_overlap: int = 300

    # np. "simple", "spacy", "tiktoken", "sentencepiece"
    tokenization_variant: str = "simple"

    # =========================
    # EMBEDDINGS
    # =========================
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_size: int = 384
    embedding_batch_size: int = 512

    # =========================
    # RETRIEVER / VECTOR STORE
    # =========================
    qdrant_url: str = "http://localhost:6333"
    base_collection_name: str = "polish_romantics_and_scientists_02_03_2026"
    top_k: int = 20

    # =========================
    # LLM / ASSISTANT
    # =========================
    groq_api_key: str | None = None
    llm_model_name: str = "llama-3.3-70b-versatile"
    llm_max_context_chars: int = 9000


    # =========================
    # DERIVED VALUES
    # =========================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    