from langchain_text_splitters import CharacterTextSplitter
from typing import List, Dict

def simple_chunk_wikipedia_article(article: Dict,
                                   chunk_size: int = 500,
                                   chunk_overlap: int = 100) -> List[Dict]:
    """
    Simple chunking of a Wikipedia article by characters, using multiple separators (paragraph, newline, end of sentence).
    Returns a list of chunks while preserving the article title and URL.
    """
    separator_regex = r"\n\n|\n|\. |\? |! "

    text_splitter = CharacterTextSplitter(
        separator=separator_regex,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=True
    )

    chunks_text = text_splitter.split_text(article["text"])

    chunks = []
    for i, ch in enumerate(chunks_text, start=1):
        chunks.append({
            "article_id": article.get("id", 0),
            "title": article["title"],
            "url": article.get("url", ""),
            "chunk_id": i,
            "chunk_text": ch
        })
    return chunks
