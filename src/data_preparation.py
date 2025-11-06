import json
import csv
import random

from datasets import load_dataset
from typing import List, Dict

from src.data_chunking_simple import simple_chunk_wikipedia_article


def download_wiki_data(dataset_name: str, limit: int = None) -> List[Dict]:
    """
    Loads a Wikipedia dataset from Hugging Face.
    """
    dataset = load_dataset(dataset_name, split="train")

    if limit:
        dataset = dataset.select(range(limit))

    articles = []
    for item in dataset:
        articles.append({
            "title": item["title"],
            "text": item["text"],
            "url": item["url"]
        })

    return articles


def save_wiki_to_json(articles: List[Dict], path: str):
    """
    Saves a list of Wikipedia articles to a JSON.

    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Saved to JSON: {path}")

def save_wiki_to_csv(articles: List[Dict], path: str):
    """
    Saves a list of Wikipedia articles to a CSV file.

    """
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text", "url"])
        writer.writeheader()
        writer.writerows(articles)
    print(f"Saved to CSV: {path}") 

def get_wikipedia_article_count(dataset_name: str) -> int:
    """
    Returns the number of articles in the specified Wikipedia dataset.
    """
    dataset = load_dataset(dataset_name, split="train")
    total = len(dataset)
    print(f"Total number of Wikipedia articles in '{dataset_name}': {total:,}")
    return total


        
