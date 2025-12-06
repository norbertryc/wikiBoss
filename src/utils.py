import json
import time
from functools import wraps
from pathlib import Path
from tqdm import tqdm

from .logging_config import logger


def file_exists(path):
    """Check if a file exists."""
    if path:
        return Path(path).exists()
    else:
        return None

def save_in_batches(batch_size: int = 10000):
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
    """"""
    def __init__(self, input_path: str, 
                 output_path: str = None,
                 clear_output: bool = False,
                 load_on_init: bool = False,
                 num_lines: int = None,
                 start_loading: int = 0
                 ):
        self.input_path = input_path
        self.output_path = output_path
        self.clear_output = clear_output
        self.articles = []

        if clear_output:
            output_file = Path(output_path)
            if file_exists(output_file):
                output_file.unlink()
                logger.info(f"File {output_file.name} removed.")
        else:
            if file_exists(self.output_path):
                logger.warning(f"File {self.output_path} already exists. New data will be appended.")

        if load_on_init:
            self.load(num_lines=num_lines, start=start_loading)

    def load(self, num_lines: int = None, start: int = 0):
        """Load articles from a jsonl file into memory."""
        count = 0

        with open(self.input_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i < start:
                    continue
                if num_lines is not None and count >= num_lines:
                    break
                try:
                    record = json.loads(line)
                    self.articles.append(record)
                    count += 1
                except json.decoder.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.articles)} articles from {self.input_path}")


def read_jsonl_record(path, line_number):
    """View the selected record from the jsonl file."""
    with open(path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == line_number:
                return json.loads(line)
    raise IndexError(f"The {path} file has fewer than {line_number} lines.")