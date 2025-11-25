from transformers import AutoTokenizer, AutoConfig
from langchain_text_splitters import (Tokenizer as TokenCountTokenizer,
                                      split_text_on_tokens,
                                      ExperimentalMarkdownSyntaxTextSplitter)

from .utils import save_in_batches, track_progress_and_time, DataLoader
from .logging import logger
from config import HUGGING_FACE_MODEL

BASE_TOKENIZER = AutoTokenizer.from_pretrained(HUGGING_FACE_MODEL, use_fast=True)
BASE_TOKENIZER_CONFIG = AutoConfig.from_pretrained(HUGGING_FACE_MODEL)
MAX_TOKENS = BASE_TOKENIZER_CONFIG.max_position_embeddings


class Chunker(DataLoader):
    """"""

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 clear_output: bool = False,
                 strategy: str = None,
                 tokenizer: AutoTokenizer = BASE_TOKENIZER):
        super().__init__(input_path, output_path, clear_output)
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.max_tokens = MAX_TOKENS
        self.splitters_by_strategy = {
            "on_tokens": self.split_simple_on_tokens,
            "on_md_headers": self.split_on_md_headers
        }

    def split_simple_on_tokens(self, article: str|dict, max_tokens_update: int = None
                               ) -> list:
        """"""
        if max_tokens_update is None:
            max_tokens = self.max_tokens
        else:
            max_tokens = max_tokens_update

        splitter = TokenCountTokenizer(tokens_per_chunk=max_tokens,
                                       chunk_overlap=int(self.max_tokens * 0.1),
                                       decode=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
                                       encode=lambda text: self.tokenizer.encode(text, add_special_tokens=False)
                                       )
        if isinstance(article, str):
            return split_text_on_tokens(text=article, tokenizer=splitter)
        elif isinstance(article, dict):
            return split_text_on_tokens(text=article["text"], tokenizer=splitter)

    def split_on_md_headers(self, article: dict) -> list:
        """"""
        splitter = ExperimentalMarkdownSyntaxTextSplitter()
        docs = splitter.split_text(f"# {article['title']}\n" + article["text"])
        chunks = []

        for doc in docs:
            heading = ""

            for k, v in doc.metadata.items():
                heading += f"{int(k[-1])*'#'} {v}\n"

            heading_tokens = len(self.tokenizer.encode(heading, add_special_tokens=False))
            doc_split_by_tokens = self.split_simple_on_tokens(doc.page_content,
                                                              max_tokens_update=self.max_tokens - heading_tokens)
            for chunk in doc_split_by_tokens:
                chunks.append(
                    {
                        "text": heading + chunk,
                        "metadata": doc.metadata,
                    }
                )
        return chunks

    @save_in_batches(batch_size=10000)
    @track_progress_and_time("Chunking")
    def chunk(self):

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

        logger.info(f"Chunked {count} articles with strategy {self.strategy}."
                    f" Saved chunks to {self.output_path}")