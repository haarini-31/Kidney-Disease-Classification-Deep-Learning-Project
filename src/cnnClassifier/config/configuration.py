from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.data_ingestion_config import DataIngestionConfig
from cnnClassifier.constants import *


class ConfigurationManager:
    def __init__(self):
        """
        Loads config.yaml and params.yaml from project root
        """

        # 🔒 Project root (ABSOLUTE, notebook-safe)
        self.project_root = Path(
            r"C:\Users\haari\OneDrive\Desktop\aiml 2026\Kidney-Disease-Classification-Deep-Learning-Project"
        )

        self.config_path = self.project_root / "configs" / "config.yaml"
        self.params_path = self.project_root / "params.yaml"

        # Safety checks
        if not self.config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {self.config_path}")

        if not self.params_path.exists():
            raise FileNotFoundError(f"params.yaml not found at {self.params_path}")

        # Load YAMLs
        self.config = read_yaml(self.config_path)
        self.params = read_yaml(self.params_path)

        # Create artifacts root directory
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns DataIngestionConfig entity
        """

        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzipped_dir=Path(config.unzipped_dir),
        )
