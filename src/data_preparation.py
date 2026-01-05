import json
import os

from config import WIKIPEDIA_PL_DATASET
from datasets import load_dataset


# =========================================================
# DOWNLOAD + STREAMING (BEZ RAM)
# =========================================================

def download_wiki_data(dataset_name: str, limit: int | None = None):
    """
    Streams Wikipedia articles from HuggingFace (no RAM explosion).
    """
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True
    )

    for i, item in enumerate(dataset):
        if limit is not None and i >= limit:
            break

        yield {
            "title": item.get("title"),
            "text": item.get("text"),
            "url": item.get("url"),
        }


def save_wiki_to_jsonl(articles, path: str):
    """
    Saves Wikipedia articles as JSONL (1 article per line).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"Saved Wikipedia to JSONL: {path}")



def load_wiki_from_jsonl(path: str):
    """
    Loads Wikipedia JSONL file line by line (generator).
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":
    RAW_WIKI_JSONL = "data/raw_wiki_pl.jsonl"

    print("Streaming Wikipedia download...")

    articles = download_wiki_data(
        WIKIPEDIA_PL_DATASET,
        limit=None  
    )

    print("Saving Wikipedia as JSONL...")
    save_wiki_to_jsonl(articles, RAW_WIKI_JSONL)

    print("DONE.")
