from datasets import load_dataset
from urllib.parse import urlparse
from tqdm import tqdm
import csv

# 1. Load the dataset
print("Loading dataset...")
dataset = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train[8000000:10000000]")


def extract_site(urls):
    if isinstance(urls, list) and len(urls) > 0:
        parsed_url = urlparse(urls[0])
        return parsed_url.netloc
    return "Unknown"


cooking_csv = 'cooking_entries.csv'
csv_file = open(cooking_csv, mode='w', newline='', encoding='utf-8')
writer = None

print("Processing entries and writing cooking.stackexchange.com entries to CSV...")
for entry in tqdm(dataset, desc="Filtering and writing cooking entries"):
    urls = entry.get('metadata', [])
    site = extract_site(urls)

    if site.lower() == "cooking.stackexchange.com":
        if writer is None:
            writer = csv.DictWriter(csv_file, fieldnames=entry.keys())
            writer.writeheader()

        writer.writerow(entry)

csv_file.close()

print(f"All cooking.stackexchange.com entries have been saved to '{cooking_csv}'.")
