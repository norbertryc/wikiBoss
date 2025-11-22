import json
import time
from functools import wraps
from pathlib import Path
from tqdm import tqdm

from .logging import logger


def save_in_batches(batch_size: int = 500):
    """"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            output_path = Path(getattr(self, "output_path", None))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            batch = []
            with open(output_path, "a", encoding="utf-8") as file:
                processed = func(self, *args, **kwargs)
                for line in processed:
                    batch.append(line)
                    if len(batch) >= batch_size:
                        for record in batch:
                            file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        batch.clear()
            return processed
        return wrapper
    return decorator


def track_progress_and_time(desc: str = "Processing"):
    """"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            results = func(self, *args, **kwargs)

            for attr in ["articles", "chunks"]:
                if hasattr(self, attr):
                    total = len(getattr(self, attr))
                    break
                else:
                    total = 0

            bar = tqdm(desc=desc, total=total, unit="items")

            def generator():
                try:
                    for line in results:
                        bar.update(1)
                        yield line
                finally:
                    bar.close()
                    total_time = time.time() - start_time
                    logger.info(f"{desc} total time: {total_time:.2f} s")

            return generator()
        return wrapper
    return decorator


class DataLoader:
    def __init__(self, input_path: str, output_path: str, clear_output: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.articles = []

        if clear_output:
            output_file = Path(self.output_path)
            if output_file.exists():
                output_file.unlink()
                logger.info(f"File {output_file.name} removed.")

    def load(self, num_lines: int = None):
        "Load articles from a jsonl file into memory."
        with open(self.input_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if num_lines is not None and i >= num_lines:
                    break
                try:
                    record = json.loads(line)
                    self.articles.append(record)
                except json.decoder.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(self.articles)} articles from {self.input_path}")