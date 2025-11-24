import json
import csv
import os

from glob import glob

from config import RAW_WIKI_JSON, WIKIPEDIA_PL_DATASET, DATA_FOLDER
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

def load_wiki_from_json(path: str):
    """
    Loads Wikipedia articles from the given JSON file.
    If the file does not exist or is empty, looks for the most recent previous JSON in DATA_FOLDER.
    Raises FileNotFoundError if no usable JSON is found.
    """
    # if current file exists and not empty, load it
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        if articles:
            print(f"Loaded {len(articles):,} articles from {path}")
            return articles
        else:
            print(f"{path} is empty, looking for previous JSON...")

   
    json_files = sorted(glob(os.path.join(DATA_FOLDER, "raw_wiki_*.json")), reverse=True)
    for file_path in json_files:
        if file_path == path:
            continue #no current file, we take older one
        with open(file_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        if articles:
            print(f"Loaded {len(articles):,} articles from {file_path}")
            return articles

    # jeśli żaden plik nie był do użycia
    raise FileNotFoundError("No usable JSON files found in DATA_FOLDER.")


 

if __name__ == "__main__":
    print("Downloading Wikipedia data...")
    articles = download_wiki_data(WIKIPEDIA_PL_DATASET, limit=3000)

    print("Saving JSON...")
    save_wiki_to_json(articles, RAW_WIKI_JSON)

    print("DONE.")


        
