import json
import time
import logging
from functools import wraps
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def file_exists(path: str | Path) -> bool:
    """
    Checks if a file exists.
    Args:
        path (str | Path): The path to check. Can be None or empty.
    Returns:
        bool: True if file exists, False otherwise (including if path is None).
    """
    return Path(path).exists() if path else False


def save_in_batches(batch_size: int = 10000):
    """
    Decorator that buffers generator output and writes to JSONL in batches.

    Requires the decorated class instance to have an 'output_path' attribute.
    Skips 'None' values yielded by the generator.

    Args:
        batch_size (int): Number of records to accumulate before writing to disk.
            Defaults to 10000.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            output_path = Path(getattr(self, "output_path", None))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            batch = []
            with open(output_path, "a", encoding="utf-8") as file:
                processed = func(self, *args, **kwargs)
                for line in processed:
                    if line is None:
                        continue
                    batch.append(line)
                    if len(batch) >= batch_size:
                        for record in batch:
                            file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        batch.clear()
                if batch:
                    for record in batch:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return wrapper
    return decorator


def track_progress_and_time(desc: str = "Processing", attr="articles"):
    """
    Decorator that adds a tqdm progress bar and logs execution time.
    Wraps a generator function, updating the progress bar as items are yielded.

    Args:
        desc (str): Description text displayed next to the progress bar.
        attr (str): Name of the class attribute (list) used to determine
            the total number of items for the progress bar. Defaults to "articles".
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):

            if hasattr(self, attr):
                total = len(getattr(self, attr))
            else:
                total = None

            start_time = time.time()
            results = func(self, *args, **kwargs)
            bar = tqdm(iterable=results, desc=desc, total=total, unit="items")

            try:
                for item in bar:
                    yield item
            finally:
                bar.close()
                total_time = time.time() - start_time
                logger.info(f"{desc} total time: {total_time:.2f} s")

        return wrapper
    return decorator


class DataLoader:
    """
    Loads JSONL records (articles) from disk into memory.

    Args:
        input_path (str, optional): Path to the input JSONL file. Set to None only for existing
                        vector store's operations by QdrantManager class.
        output_path (str, optional): Path for downstream output file (e.g. cleaned JSONL).
                        If file exists, new data is appended unless clear_output is True.
                        Set to None only for QdrantManager class.
        clear_output (bool): If True, deletes existing output_path before processing. Defaults to False.
        load_on_init (bool): If True, automatically calls load() on initialization. Defaults to False.
        num_lines (int, optional): Max records to load if load_on_init is True.
        start_loading (int): Number of initial lines to skip if load_on_init is True. Defaults to 0.

    Attributes:
        articles (list): List of loaded article dictionaries.
    """
    def __init__(
            self,
            input_path: str = None,
            output_path: str = None,
            clear_output: bool = False,
            load_on_init: bool = True,
            num_lines: int = None,
            start_loading: int = 0
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.clear_output = clear_output
        self.articles = []

        if clear_output and self.output_path:
            if file_exists(self.output_path):
                Path(self.output_path).unlink()
                logger.info(f"File {Path(self.output_path).name} removed.")
        elif file_exists(self.output_path):
            logger.warning(f"File {self.output_path} already exists. New data will be appended.")

        if load_on_init and input_path:
            self.load(num_lines=num_lines, start=start_loading)

    def load(self, num_lines: int = None, start: int = 0):
        """
        Reads records from input_path into self.articles list.

        Args:
            num_lines (int, optional): Max number of records to load. If None, reads entire file.
            start (int): Number of lines to skip from the beginning of file. Defaults to 0.
        """
        total_records = count_jsonl_records(self.input_path)

        # validate start parameter
        if start >= total_records:
            if num_lines is not None:
                # Load last num_lines records instead
                new_start = max(0, total_records - num_lines)
                logger.warning(
                    f"Start position ({start}) >= total records ({total_records}). "
                    f"Loading last {min(num_lines, total_records)} records instead (from position {new_start})."
                )
                start = new_start
            else:
                logger.error(f"Start position ({start}) >= total records ({total_records}). Nothing to load.")
                return

        if start < 0:
            logger.warning(f"Start position ({start}) is negative. Setting to 0.")
            start = 0

        # validate num_lines parameter
        max_available = total_records - start

        if num_lines is not None:
            if num_lines <= 0:
                logger.error(f"num_lines ({num_lines}) must be positive. Nothing to load.")
                return

            if num_lines > max_available:
                logger.warning(
                    f"Requested {num_lines} records from position {start}, "
                    f"but only {max_available} available. Will load {max_available} records."
                )
                num_lines = max_available

        # load records
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
                    logger.warning(f"Skipping invalid JSON at line {i + 1}")
                    continue

        logger.info(f"Loaded {len(self.articles)} articles from {self.input_path} "
                    f"(total available: {total_records})")


def read_jsonl_record(path: str,
                      line_number: int = None,
                      wiki_id: int | str = None,
                      title: str = None
                      ) -> dict:
    """
    Retrieves a single record from a JSONL file based on line number, ID, or title.

    Useful for inspecting intermediate results of data processing pipelines (e.g., debugging
    cleaned articles or generated chunks).

    Args:
        path (str): Path to the JSONL file.
        line_number (int, optional): Line index (0-based) to retrieve directly.
        wiki_id (int | str, optional): Wikipedia ID to search for.
        title (str, optional): Article title to search for.
    Returns:
        dict: The found record as a dictionary.
    Raises:
        ValueError: If not exactly one lookup criterion is provided.
        IndexError: If the specified line_number does not exist.
        KeyError: If no record matches the given wiki_id or title.
        FileNotFoundError: If the file does not exist.
    """
    criteria_count = sum(arg is not None for arg in [line_number, wiki_id, title])
    if criteria_count != 1:
        raise ValueError(f"Provide exactly one lookup criterion. You provided {criteria_count}.")

    with open(path, "r", encoding="utf-8") as file:

        if line_number is not None:
            for i, line in enumerate(file):
                if i == line_number:
                    return json.loads(line)
            raise IndexError(f"Line {line_number} not found in {path}")

        target_key = "id" if wiki_id is not None else "title"
        target_val = wiki_id if wiki_id is not None else title

        for line in file:
            try:
                record = json.loads(line)
                record_val = record.get(target_key)

                if str(record_val) == str(target_val):
                    return record

            except json.JSONDecodeError:
                continue

    raise KeyError(f"Record with {target_key}='{target_val}' not found in {path}")


def count_jsonl_records(file_path: str) -> int:
    """
    Record count for JSONL file.

    Args:
        file_path (str): Path to JSONL file
    Returns:
        int: Number of non-empty lines in file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())
