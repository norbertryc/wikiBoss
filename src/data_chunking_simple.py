from langchain_text_splitters import CharacterTextSplitter
from typing import List, Dict

def simple_chunk_wikipedia_article(article: Dict,
                                   chunk_size: int = 500,
                                   chunk_overlap: int = 100) -> List[Dict]:

    # Separate by paragraphs first
    paragraphs = article["text"].split("\n\n")

    # 2) Split each paragraph into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",       # split by new lines in paragraphs
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=False
    )

    chunks = []
    chunk_id = 1

    for para in paragraphs:
        small_chunks = text_splitter.split_text(para)
        for ch in small_chunks:
            chunks.append({
                "article_id": article.get("id", 0),
                "title": article["title"],
                "url": article.get("url", ""),
                "chunk_id": chunk_id,
                "chunk_text": ch
            })
            chunk_id += 1

    return chunks

