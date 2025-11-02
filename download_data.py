import pathlib
from beir import util
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Downloads the SciFact dataset from the BEIR benchmark.
    """
    dataset = "scifact"
    out_dir = pathlib.Path("datasets")
    
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    
    data_path = util.download_and_unzip(url, str(out_dir))
    
    logger.info(f"Dataset downloaded and unzipped to: {data_path}")

if __name__ == "__main__":
    main()
