"""Global configuration."""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class EmbeddingModelConfig(BaseSettings):
    """Embedding model configuration."""
    model: str = Field(
        default="intfloat/multilingual-e5-base",
        description="HuggingFace model identifier",
    )
    is_sentence_transformers_model: bool = Field(
        default=True,
        description="Whether model is native sentence_transformers model",
    )


class QdrantConfig(BaseSettings):
    """"Qdrant vector database connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)


class HNSWConfig(BaseSettings):
    """HNSW indexing parameters for Qdrant."""
    m: int = Field(
        default=32,
        ge=4,
        le=64,
        description=(
            "Bi-directional links per element. "
            "32 is optimal for 768D embeddings (e.g., multilingual-e5-base) "
            "and provides better recall for large collections (2-5M vectors). "
            "Use 16 for prototyping or lower-dimensional data or low RAM."
        )
    )
    ef_construct: int = Field(
        default=200,
        ge=100,
        le=512,
        description=(
            "Size of dynamic candidate list during index construction. "
            "Higher = better quality, slower indexing"
        )
    )


class Config(BaseSettings):
    """Global RAG system configuration. Loads from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    # Groq api key
    groq_api_key: str = Field(
        default=None,
        min_length=20,
        description="Groq API key for LLM inference (required only for RAG app)"
    )
    def require_groq_key(self) -> str:
        """Validate that Groq API key is set. Call this in RAG app."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required for RAG query functionality. "
                "Set it in .env file."
            )
        return self.groq_api_key

    # Chat models
    primary_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Primary LLM model for generation"
    )
    fallback_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Fallback model when primary fails"
    )

    # Directory paths (computed from project root)
    @computed_field
    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return Path(__file__).parent

    @computed_field
    @property
    def data_dir(self) -> Path:
        """Main data directory."""
        return self.project_root / "data"

    @computed_field
    @property
    def raw_dir(self) -> Path:
        """Raw data directory."""
        path = self.data_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def processed_dir(self) -> Path:
        """Processed data directory."""
        path = self.data_dir / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def chunked_dir(self) -> Path:
        """Chunked data directory."""
        path = self.processed_dir / "chunked"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # File paths
    dump_file: str = Field(
        default="plwiki-latest-pages-articles.xml.bz2",
        description="Wikipedia dump filename"
    )

    @computed_field
    @property
    def dump_path(self) -> Path:
        """Full path to Wikipedia dump."""
        return self.raw_dir / self.dump_file

    @computed_field
    @property
    def dump_url(self) -> str:
        """URL to download Wikipedia dump."""
        return f"https://dumps.wikimedia.org/plwiki/latest/{self.dump_file}"

    @computed_field
    @property
    def parsed_jsonl(self) -> Path:
        """Path to parsed articles JSONL."""
        return self.raw_dir / "parsed_articles.jsonl"

    @computed_field
    @property
    def cleaned_jsonl(self) -> Path:
        """Path to cleaned articles JSONL."""
        return self.processed_dir / "cleaned_articles.jsonl"

    @computed_field
    @property
    def log_file(self) -> Path:
        """Path to log file."""
        return self.project_root / "logs.txt"

    # Nested configs
    embedding_model: EmbeddingModelConfig = Field(
        default_factory=EmbeddingModelConfig
    )
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)

    # Configured storage name
    storage_name_suffix_override: str = Field(
        default="",
        description="Manual suffix override. If empty, auto-generated from embedding model name"
    )

    @computed_field
    @property
    def storage_name_suffix(self) -> str:
        """Auto-generated suffix from embedding model name."""
        model_name = self.embedding_model.model
        suffix = model_name.split("/")[-1] if "/" in model_name else model_name
        return f"_{suffix}"

    # Naming helpers
    def get_chunked_jsonl_path(
            self,
            chunking_method: str,
            suffix_override: str | None = None
    ) -> Path:
        """Get chunked JSONL path with optional manual suffix override."""
        suffix = f"_{suffix_override}" if suffix_override else self.storage_name_suffix
        filename = f"{chunking_method}_chunks{suffix}.jsonl"
        return self.chunked_dir / filename

    def get_collection_name(
            self,
            chunking_method: str,
            suffix_override: str | None = None
    ) -> str:
        """Get collection name with optional manual suffix override."""
        suffix = f"_{suffix_override}" if suffix_override else self.storage_name_suffix
        return f"{chunking_method}{suffix}"

    # Hardware & performance settings
    device: str = Field(
        default="cuda",
        description="Device for embedding model: 'cuda' or 'cpu'"
    )

    half_precision: bool = Field(
        default=False,
        description="Use FP16 for embedding model on GPU (saves time and memory)"
    )

    encoding_batch_size: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Batch size for embedding generation (adjust for your GPU memory)"
    )

    upload_batch_size: int = Field(
        default=10000,
        ge=100,
        le=50000,
        description="Number of points to accumulate before Qdrant upload"
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    @computed_field
    @property
    def log_dir(self) -> Path:
        """Logs directory."""
        path = self.project_root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field
    @property
    def app_log_file(self) -> Path:
        """Main application log file."""
        return self.log_dir / "app.log"

    def __repr__(self) -> str:
        return f"Config(primary_model={self.primary_model}, qdrant={self.qdrant.host}:{self.qdrant.port})"


config = Config()
