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
