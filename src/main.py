import config

from src.data_preparation import download_wiki_data, save_wiki_to_json, save_wiki_to_csv

#Load raw wikipedia data
raw_data = download_wiki_data(dataset_name=config.WIKIPEDIA_PL_DATASET, limit=15)

#Save raw data to JSON file
save_wiki_to_json(articles=raw_data, path=config.RAW_WIKI_JSON)

#Save raw data to CSV file
save_wiki_to_csv(articles=raw_data, path=config.RAW_WIKI_CSV)

print("Number of articles loaded:", len(raw_data))
print("Selected article title:", raw_data[7]["title"])
print("Selected article URL:", raw_data[7]["url"])

if __name__ == "__main__":
    pass