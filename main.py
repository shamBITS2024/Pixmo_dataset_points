import os
import asyncio
import aiohttp
import logging
from hashlib import sha256
from datasets import load_dataset
import requests

# Configuration
OUTPUT_DIR = "/changed_data/image"
CHECKPOINT_FILE = "completed_images_metadata.txt"
FAILED_FILE = "unaccessible_url.txt"
HASH_DIFF_FILE = "hash_different.txt"
BATCH_SIZE = 20
TIMEOUT = 10  # Per request timeout
MAX_RETRIES = 3  # Maximum retry attempts for transient issues

# Dataset parameters
DATASET_NAME = "allenai/pixmo-points"
SPLIT = "train"

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
data = load_dataset(DATASET_NAME, split=SPLIT)
completed_hashes = set()

for example in data:
    url = example["image_url"]
    print(url)
    expected_hash = example["image_sha256"]
    
    for attempt in range(MAX_RETRIES):
        try:
            if expected_hash in completed_hashes:
                logging.info(f"SKIPPED: {url}")
                break
                
            with requests.get(url, timeout=TIMEOUT) as response:
                content = response.content
                
            computed_hash = sha256(content).hexdigest()
            
            if computed_hash != expected_hash:
                with open(HASH_DIFF_FILE, "a") as f:
                    f.write(f"URL: {url}, Expected hash: {expected_hash}, Computed hash: {computed_hash}\n")
                logging.warning(f"Hash mismatch for URL: {url}")
                raise ValueError("Hash mismatch! Possible corrupted or incorrect file.")
            else: 
                completed_hashes.add(expected_hash)   
                output_path = os.path.join(OUTPUT_DIR, f"{computed_hash}.jpg")
                with open(output_path, "wb") as f:
                    f.write(content)
                with open(CHECKPOINT_FILE, "a") as checkpoint_handle:    
                    checkpoint_handle.write(f"{url},{computed_hash}\n")
                    checkpoint_handle.flush()
                    completed_hashes.add(expected_hash)
                    logging.info(f"SUCCESS: {url}")
                break
            
        except Exception as e:
            print(e)
