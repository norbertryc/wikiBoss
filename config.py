from datetime import datetime

WIKIPEDIA_PL_DATASET = "chrisociepa/wikipedia-pl-20230401"

DATA_FOLDER = "data"

TODAY = datetime.now().strftime("%Y%m%d")

RAW_WIKI_JSON = f"{DATA_FOLDER}/raw_wiki_{TODAY}.json"
RAW_WIKI_CSV = f"{DATA_FOLDER}/raw_wiki_{TODAY}.csv"