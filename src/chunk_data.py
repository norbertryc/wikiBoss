from langchain_text_splitters import (Tokenizer,
                                      split_text_on_tokens,
                                      ExperimentalMarkdownSyntaxTextSplitter)

from .utils import save_in_batches, track_progress_and_time, DataLoader, count_jsonl_records
from .logging_config import logger
from .embedding_engine import EmbeddingEngine


class Chunker(DataLoader):
    """
    Splits Wikipedia articles into embedding-ready chunks using specified strategy.

    Args:
        embedding_engine (EmbeddingEngine): Engine providing tokenizer and model config
        strategy (str): Chunking strategy - 'on_tokens' or 'on_md_headers'
        input_path (str): Path to input JSONL file with raw articles (required)
        output_path (str): Path to output JSONL file for cleaned articles (required)
        *args, **kwargs: Passed to DataLoader parent class

    Attributes:
        strategy (str): Active chunking strategy
        engine (EmbeddingEngine): Embedding engine instance
        max_tokens (int): Maximum tokens per chunk from embedding model
        chunk_prefix (str): Model-specific prefix (e.g., "passage: " for E5)
        splitters_by_strategy (dict): Maps strategy names to splitter methods
    """

    def __init__(self,
                 *args,
                 input_path: str = None,
                 output_path: str = None,
                 embedding_engine: EmbeddingEngine,
                 strategy: str = None,
                 **kwargs):

        if input_path is None:
            raise ValueError("Chunker requires 'input_path' to be specified")
        if output_path is None:
            raise ValueError("Chunker requires 'output_path' to be specified")

        super().__init__(*args, input_path=input_path, output_path=output_path, **kwargs)
        self.strategy = strategy
        self.engine = embedding_engine
        self.chunk_prefix = embedding_engine.chunk_prefix
        self.max_tokens = embedding_engine.embedding_size

        self.splitters_by_strategy = {
            "on_tokens": self.split_on_tokens_strategy,
            "on_md_headers": self.split_on_md_headers
        }

    def _tokenize_and_split(self, text: str, content_budget: int, overlap_ratio: float = 0.2) -> list[str]:
        """
        Splits text into token-based chunks with overlap.

        Args:
            text (str): Text content to split
            content_budget (int): Maximum tokens per chunk
            overlap_ratio (float): Ratio of overlap between consecutive chunks (default: 0.2)

        Returns:
            list[str]: List of text chunks without formatting
        """
        token_count = self.engine.get_token_count(text)

        if token_count <= content_budget:
            return [text]

        # calculate overlap
        calculated_overlap = int(content_budget * overlap_ratio)
        final_overlap = 0 if calculated_overlap >= content_budget else calculated_overlap

        splitter = Tokenizer(
            tokens_per_chunk=content_budget,
            chunk_overlap=final_overlap,
            decode=lambda ids: self.engine.tokenizer.decode(ids, skip_special_tokens=True),
            encode=lambda text: self.engine.tokenizer.encode(text, add_special_tokens=False)
        )

        return split_text_on_tokens(text=text, tokenizer=splitter)

    def split_on_tokens_strategy(self, article: dict) -> list[dict]:
        """
        Splits article into token-based chunks with title header and prefix.
        Each chunk includes: prefix + "# title\n" + content.

        Args:
            article (dict): Article with 'title' and 'text' keys

        Returns:
            list[dict]: Chunks with 'text' (formatted) and 'metadata' keys
        """
        title_header = f"# {article['title']}\n"
        prefix_and_title = self.chunk_prefix + title_header

        # reserve tokens for prefix and title
        prefix_title_tokens = self.engine.get_token_count(prefix_and_title)
        content_budget = self.max_tokens - prefix_title_tokens

        text_chunks = self._tokenize_and_split(article["text"], content_budget, overlap_ratio=0.2)

        return [
            {
                "text": prefix_and_title + chunk,
                "metadata": {"Header 1": article["title"]}
            }
            for chunk in text_chunks
        ]

    def split_on_md_headers(self, article: dict) -> list[dict]:
        """
        Splits article by markdown headers, then by tokens within each section.
        Each chunk includes: prefix + full header hierarchy + content.

        Args:
            article (dict): Article with 'title' and 'text' keys

        Returns:
            list[dict]: Chunks with 'text' (formatted) and 'metadata' (header hierarchy) keys
        """
        splitter = ExperimentalMarkdownSyntaxTextSplitter()
        docs = splitter.split_text(f"# {article['title']}\n" + article["text"])
        chunks = []

        for doc in docs:
            # reconstruct header hierarchy from metadata
            heading = "".join(f"{int(k[-1]) * '#'} {v}\n" for k, v in doc.metadata.items())

            # reserve tokens for prefix and heading
            heading_tokens = self.engine.get_token_count(self.chunk_prefix + heading)
            content_budget = self.max_tokens - heading_tokens

            # split section content by tokens
            text_chunks = self._tokenize_and_split(doc.page_content, content_budget, overlap_ratio=0.1)

            for chunk in text_chunks:
                if chunk.strip():
                    chunks.append({
                        "text": self.chunk_prefix + heading + chunk,
                        "metadata": doc.metadata,
                    })

        return chunks

    @save_in_batches(batch_size=20000)
    @track_progress_and_time("Chunking")
    def chunk(self):
        """
        Main chunking pipeline. Processes articles using configured strategy.
        Yields articles with 'chunks' field, original 'text' removed.

        Yields:
            dict: Processed article with chunks or None if error occurred
        """
        logger.info(f"Start chunking with strategy {self.strategy}...")

        count = 0

        for i, article in enumerate(self.articles):
            try:
                splitter = self.splitters_by_strategy.get(self.strategy)

                article["chunks"] = splitter(article)
                del article["text"]

                count += 1
                yield article

            except Exception as e:
                logger.error(f"The article no {i} (wiki id {article["id"]}): {article["title"]} skipped"
                             f" because of exception:\n'{e}'")
                yield None

        logger.info(f"Chunked {count} articles with strategy '{self.strategy}'."
                    f" Saved chunks to {self.output_path} (total records: {count_jsonl_records(self.output_path)}).")
