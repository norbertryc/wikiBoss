from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    # =========================
    # DATA
    # =========================

    raw_wiki_jsonl: Path = Path("data/raw_wiki_pl.jsonl")

    # =========================
    # CHUNKING
    # =========================
    chunk_size: int = 500
    chunk_overlap: int = 100

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
    base_collection_name: str = "wiki_chunks"
    top_k: int = 20

    # =========================
    # LLM / ASSISTANT
    # =========================
    groq_api_key: str 
    llm_model_name: str = "llama-3.3-70b-versatile"
    llm_top_k: int = 5
    llm_max_context_chars: int = 12000
   

    # =========================
    # DERIVED VALUES
    # =========================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    