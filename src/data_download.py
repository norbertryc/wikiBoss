import json
from pathlib import Path
from config import Config
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm 


# =========================================================
# CONSTANTS AND PATHS
# =========================================================
WIKIPEDIA_PL_DATASET = "chrisociepa/wikipedia-pl-20230401"
DATA_FOLDER = Path("data")


# =========================================================
# DOWNLOAD + STREAMING WIKIPEDIA DATA
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

def download_specific_titles(titles):
    dataset = load_dataset(
        WIKIPEDIA_PL_DATASET,
        split="train",
        streaming=True
    )

    titles_set = set(titles)

    for item in dataset:
        if item["title"] in titles_set:
            yield {
                "title": item["title"],
                "text": item["text"],
                "url": item["url"],
            }
            titles_set.remove(item["title"])
            if not titles_set:
                break


def save_wiki_to_jsonl(articles, path: str):
    """
    Saves Wikipedia articles as JSONL (1 article per line).
    """

    with open(path, "w", encoding="utf-8") as f:
        for article in tqdm(articles, desc="Saving articles"):
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"Saved Wikipedia to JSONL: {path}")



def load_wiki_from_jsonl(path: str):
    """
    Loads Wikipedia JSONL file line by line (generator).
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# if __name__ == "__main__":
    # today_str = datetime.now().strftime("%Y%m%d")
    # raw_wiki_jsonl = DATA_FOLDER / f"raw_wiki_pl_JSONL_format_{today_str}.jsonl"
    # DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # print("Streaming Wikipedia download...")

    # articles = download_wiki_data(
    #     WIKIPEDIA_PL_DATASET, limit=None)

    # print("Saving Wikipedia as JSONL...")
    # save_wiki_to_jsonl(articles, raw_wiki_jsonl)

    # print("DONE.")

    # # =============================
    # # CHECK OLD AND NEW JSONL FILES
    # # =============================
    # old_file = Path("data/raw_wiki_pl.jsonl")
    # new_file = raw_wiki_jsonl 

    # print("OLD file exists:", old_file.exists(), "size:", old_file.stat().st_size)
    # print("NEW file exists:", new_file.exists(), "size:", new_file.stat().st_size)

if __name__ == "__main__":

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    # New file for selected titles
    selected_file = DATA_FOLDER / "raw_wiki_polish_romantics_and_scientists.jsonl"

    print("Downloading selected Wikipedia titles...")
    articles = download_specific_titles(cfg.target_titles)

    print("Saving selected titles to JSONL...")
    save_wiki_to_jsonl(articles, selected_file)

    print("DONE. Saved:", selected_file)


