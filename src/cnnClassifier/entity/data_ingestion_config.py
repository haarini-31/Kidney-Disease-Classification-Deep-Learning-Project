from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    source_URL: str
    local_data_file: Path
    unzipped_dir: Path
