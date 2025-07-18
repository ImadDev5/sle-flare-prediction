import requests
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('string_download.log'),
        logging.StreamHandler()
    ]
)

# --- Configuration ---
# URL for the STRING protein info file (for Homo sapiens)
URL = 'https://stringdb-static.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz'

# Directory to save the downloaded file
DOWNLOAD_DIR = r'D:\SLE_data\raw\STRING'

# Output filename
FILE_NAME = 'protein.info.v12.0.txt.gz'

def download_file():
    """Downloads the STRING protein info file."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    file_path = os.path.join(DOWNLOAD_DIR, FILE_NAME)

    if os.path.exists(file_path):
        logging.info(f"File '{file_path}' already exists. Skipping download.")
        return

    logging.info(f"Downloading {URL} to {file_path}...")
    try:
        with requests.get(URL, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info("Download complete.")
    except requests.RequestException as e:
        logging.error(f"Failed to download file: {e}")
        # Clean up partially downloaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

if __name__ == "__main__":
    download_file()