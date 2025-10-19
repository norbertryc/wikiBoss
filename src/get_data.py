import os
import json
import bz2
import requests
import mwxml
from tqdm import tqdm

from config import DUMP_PATH, DUMP_URL, JSON_RAW


def download_wikidump(url: str, filepath: str
                     ) -> None:
    
    if os.path.exists(filepath):
        print(f"File {filepath} exists.")
        return

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        size = int(r.headers.get("Content-Length", 0))

        with open(filepath, "wb") as f, tqdm(
            total=size, unit="B", unit_scale=True, desc="Downloading wiki dump") as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def parse_dump(dump_path: str, output_path: str) -> None:

    if os.path.exists(output_path):
        print(f"File {output_path} exists.")
        return
    
    with bz2.open(dump_path, "rb") as f:
        dump = mwxml.Dump.from_file(f)
    
        with open(output_path, "w", encoding="utf-8") as out:
            for page in tqdm(dump, desc="Parsing to jsonl", unit="pages"):
    
                revisions = list(page)
                text = revisions[-1].text
                if not revisions or not text:
                    continue
        
                record = {
                    "id": page.id,
                    "title": page.title,
                    "url": f"https://pl.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
                    "text": text.strip(),
                }
        
                out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    download_wikidump(DUMP_URL, DUMP_PATH)
    parse_dump(DUMP_PATH, JSON_RAW)