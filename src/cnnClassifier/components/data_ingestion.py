import os
import gdown
import zipfile
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.data_ingestion_config import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        try:
            dataset_url = self.config.source_URL
            zip_path = Path(self.config.local_data_file)

            os.makedirs(zip_path.parent, exist_ok=True)

            logger.info(f"Downloading data from {dataset_url}")
            logger.info(f"Saving zip to: {zip_path}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?export=download&id="
            gdown.download(prefix + file_id, str(zip_path), quiet=False)

            logger.info(f"Downloaded file size: {get_size(zip_path)}")

            return zip_path

        except Exception as e:
            raise e

    def extract_zip_file(self):
        try:
            zip_path = Path(self.config.local_data_file)
            unzip_path = Path(self.config.unzipped_dir)

            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Extracting zip file: {zip_path}")

            with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
                zip_ref.extractall(str(unzip_path))

            logger.info("Zip file extracted successfully")

        except Exception as e:
            raise e
