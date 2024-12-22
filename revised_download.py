import os
import asyncio
import aiohttp
import logging
from hashlib import sha256
from datasets import load_dataset

# Configuration
OUTPUT_DIR = "/changed_data/image"
CHECKPOINT_FILE = "completed_images_metadata.txt"
FAILED_FILE = "Failed_to_access.txt"
HASH_DIFF_FILE = "hash_different.txt"
PROCESSED_FILE = "processed_images.txt"
BATCH_SIZE = 20
TIMEOUT = 10  # Per request timeout
MAX_RETRIES = 3  # Maximum retry attempts for transient issues

# Dataset parameters
DATASET_NAME = "allenai/pixmo-points"
SPLIT = "train"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
data = load_dataset(DATASET_NAME, split=SPLIT)

async def download_image(example, session, completed_url, checkpoint_handle, failed_handle,processed_handle):
    url = example["image_url"]
    # print(url)
    expected_hash = example["image_sha256"]
    
    for attempt in range(MAX_RETRIES):
        try:
            if url in completed_url:
                logging.info(f"SKIPPED: {url}")
                return
            processed_handle.write(f"{url}\n")     
            async with session.get(url, timeout=TIMEOUT) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                content = await response.read()
                
            computed_hash = sha256(content).hexdigest()
            
            if computed_hash != expected_hash:
                with open(HASH_DIFF_FILE, "a") as f:
                    f.write(f"URL: {url}, Expected hash: {expected_hash}, Computed hash: {computed_hash}\n")
                logging.warning(f"Hash mismatch for URL: {url}")
                raise ValueError("Hash mismatch! Possible corrupted or incorrect file.")
                
            output_path = os.path.join(OUTPUT_DIR, f"{computed_hash}.jpg")
            with open(output_path, "wb") as f:
                f.write(content)
                
            checkpoint_handle.write(f"{url}:::::{expected_hash}:{computed_hash}\n")
            checkpoint_handle.flush()
            completed_url.add(url)
            logging.info(f"SUCCESS: {url}")
            return
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            failed_handle.write(f"{url}\n")
            failed_handle.flush()
            logging.error(f"FAILED: {url} - {str(e)}")
            return

async def download_all_images(data):
    completed_url = set()
    
    # Open file handles outside of the async context
    checkpoint_handle = open(CHECKPOINT_FILE, "a")
    failed_handle = open(FAILED_FILE, "a")
    processed_handle= open(PROCESSED_FILE, "a")
    
    try:
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(BATCH_SIZE)
            
            async def safe_download(example):
                async with semaphore:
                    await download_image(example, session, completed_url, checkpoint_handle, failed_handle,processed_handle)
            
            # Only process the first batch
            # first_batch = list(data)[:BATCH_SIZE]
            tasks = [safe_download(example) for example in data]
            await asyncio.gather(*tasks)
            
            # logging.info(f"Completed first batch of {BATCH_SIZE} images. Stopping as requested.")
    finally:
        # Ensure files are closed properly
        checkpoint_handle.close()
        failed_handle.close()
        processed_handle.close()

if __name__ == "__main__":
    asyncio.run(download_all_images(data))
    logging.info("Test batch downloading completed.")