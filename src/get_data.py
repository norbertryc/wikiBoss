import json
import time
import bz2
import requests
import mwxml
from tqdm import tqdm

from .logging_config import logger
from .utils import file_exists


def download_wikidump(url: str, filepath: str
                     ) -> None:
    "Download the official Wikipedia dump from `url` to `filepath` if not already present."
    if file_exists(filepath):
        logger.info(f"File {filepath} exists.")
        return

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        size = int(r.headers.get("Content-Length", 0))

        with open(filepath, "wb") as f, tqdm(
            total=size, unit="B", unit_scale=True, desc="Downloading wiki dump") as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    logger.info(f"File {filepath} from {url} downloaded.")


def parse_dump(dump_path: str, output_path: str) -> None:
    """
    Parse a Wikipedia XML dump and save to a JSONL file with fields: id, title, url, text.
    The 'text' field preserves the original, unprocessed MediaWiki markup.
    If the output file already exists, the function skips parsing.
    """
    if file_exists(output_path):
        logger.info(f"File {output_path} exists.")
        return
    
    with bz2.open(dump_path, "rb") as f:
        dump = mwxml.Dump.from_file(f)

        start = time.time()
        count = 0
    
        with open(output_path, "w", encoding="utf-8") as out:
            for page in tqdm(dump, desc="Parsing to jsonl", unit="pages"):
    
                revisions = list(page)
                text = revisions[-1].text
                if not revisions or not text or text[:10].strip().lower().startswith(
                        ("#redirect", "redirect", "#patrz", "patrz")):
                    continue

                record = {
                    "id": page.id,
                    "title": page.title,
                    "url": f"https://pl.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
                    "text": text.strip(),
                }
        
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"File {output_path} from {dump_path} saved with {count} records"
                f" in {round((time.time() - start)/60, 2)} minutes.")
