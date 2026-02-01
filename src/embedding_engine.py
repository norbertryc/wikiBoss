from torch import Tensor, cuda
from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from .logging_config import logger


class EmbeddingEngine:
    """
    Wrapper for SentenceTransformers and HuggingFace models. Manages model loading, tokenizer,
    and embedding generation.

    Args:
        model_config (dict): Dictionary with 'model' key (hugging face model name/path) and optional
                            'is_sentence_transformers_model' boolean flag
        device (str): Device for model inference - 'cuda' or 'cpu' (default: 'cuda')
        use_fp16 (bool): If True, converts model to half precision on GPU (default: False)

    Attributes:
        model_name (str): Name or path of the loaded model
        is_st_model (bool): Whether model is native SentenceTransformers format
        device (str): Active device (cuda/cpu)
        tokenizer (AutoTokenizer): HuggingFace tokenizer instance
        max_seq_length (int): Maximum sequence length from model config
        model (SentenceTransformer): Loaded embedding model
    """

    def __init__(self,
                 model_config: dict,
                 device: str = "cuda",
                 use_fp16: bool = False):
        self.model_name = model_config["model"]
        self.is_st_model = model_config["is_sentence_transformers_model"]
        self.device = device if cuda.is_available() else "cpu"

        logger.info(f"Initializing EmbeddingEngine with: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        config = AutoConfig.from_pretrained(self.model_name)
        self.max_seq_length = config.max_position_embeddings

        self.chunk_prefix = ("passage: " if "e5" in self.model_name else "")
        self.query_prefix = ("query: " if "e5" in self.model_name
                             else "zapytanie: " if "mmlw" in self.model_name
                             else "")

        self.model = self._load_model()
        self.embedding_size = self.model.get_sentence_embedding_dimension()

        if use_fp16 and self.device == "cuda":
            self.model.half()
            logger.info("Model converted to fp16 (half precision)")

    def _load_model(self) -> SentenceTransformer:
        """
        Loads model as SentenceTransformer instance.
        Builds custom architecture if not native SentenceTransformers format.

        Returns:
            SentenceTransformer: Ready-to-use embedding model
        """
        if self.is_st_model:
            return SentenceTransformer(self.model_name, device=self.device)
        else:
            word_embedding_model = Transformer(self.model_name, max_seq_length=self.max_seq_length)
            pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
            return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

    def encode(self, texts: str|list[str], batch_size: int = 32) -> Tensor:
        """
        Generates embeddings for input text(s).

        Args:
            texts (str | list[str]): Single text string or list of texts to embed
            batch_size (int): Number of texts to process in parallel (default: 32)

        Returns:
            Tensor: Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        return self.model.encode(texts, batch_size=batch_size, device=self.device)

    def get_token_count(self, text: str) -> int:
        """
        Counts tokens in text without special tokens.

        Args:
            text (str): Input text to tokenize

        Returns:
            int: Number of tokens (excluding special tokens)
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
