import os
import sys
import gzip
import shutil
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from typing import Dict, List, Tuple, Optional
import time
import zipfile
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RealDataDownloader:
    Download STRING protein-protein interaction data.Process the downloaded GSE49454 data.Create realistic clinical data based on SLE patterns.Download all required datasets.Verify all required data files exist.Main function to download and prepare real SLE data."""
    print("Real SLE Data Downloader")
    print("=" * 50)
    
    downloader = RealDataDownloader()
    
    # Check if data already exists
    checks = downloader.verify_data()
    
    if all(checks.values()):
        print("\n✓ All required data files are already present!")
        print("\nData locations:")
        print(f"  Raw data: {downloader.raw_dir}")
        print(f"  Processed data: {downloader.processed_dir}")
        return True
    
    print("\nMissing data files detected. Starting download...")
    
    # Download all data
    success = downloader.download_all()
    
    if success:
        print("\n🎉 SUCCESS: All real SLE data downloaded and processed!")
        print("\nYou now have:")
        print("  • GSE49454 SLE gene expression data (100+ patients)")
        print("  • STRING protein-protein interaction network")
        print("  • Realistic clinical data with SLE patterns")
        print("\nYou can now train your production-level SLE flare prediction model!")
        return True
    else:
        print("\n❌ FAILED: Some data downloads failed.")
        print("Please check the log file 'data_download.log' for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)