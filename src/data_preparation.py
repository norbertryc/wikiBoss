import json
import csv

from datasets import load_dataset
from typing import List, Dict

def download_wiki_data(dataset_name: str, limit: int = None) -> List[Dict]:
    """
    Loads a Wikipedia dataset from Hugging Face.

    Args:
        dataset_name (str): Hugging Face dataset identifier.
        limit (int, optional): Maximum number of articles to load. If None, loads all.

    Returns:
        List[Dict]: List of articles, each as a dictionary with keys {title, text, url}
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

    Args:
        articles (List[Dict]): List of articles, each as a dictionary with keys {title, text, url}.
        path (str, optional): File path to save the JSON.

    Returns:
        None
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Saved to JSON: {path}")