import os
from pathlib import Path

from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.data_ingestion_config import EvaluationConfig
from cnnClassifier.entity.data_ingestion_config import DataIngestionConfig, TrainingConfig
from cnnClassifier.entity.data_ingestion_config import PrepareBaseModelConfig
from cnnClassifier.constants import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzipped_dir=Path(config.unzipped_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        PrepareBaseModelConfig from config.yaml + params.yaml
        """
        config = self.config.prepare_base_model
        params = self.params

        create_directories([Path(config.root_dir)])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),

            params_image_size=params.image_size,
            params_learning_rate=params.learning_rate,
            params_include_top=params.include_top,
            params_weights=params.weights,
            params_classes=params.classes,
        )

    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training
        params = self.params
        
        updated_base_model_path = Path(training_config.updated_base_model_path)
        
        # The dataset unzips into a nested folder structure
        training_data = Path(training_config.training_data) / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        
        create_directories([Path(training_config.root_dir)])

        return TrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model_path=updated_base_model_path, 
            training_data=training_data,
            params_epochs=params.epochs,
            params_batch_size=params.batch_size,
            params_is_augmentation=params.is_augmentation, 
            params_image_size=list(params.image_size)
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation_config = EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            training_data=Path(self.config.training.training_data) / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
            all_params=self.params,
            mlflow_uri="https://dagshub.com/haarini-31/Kidney-Disease-Classification-Deep-Learning-Project.mlflow",
            params_image_size=self.params.image_size,
            params_batch_size=self.params.batch_size
        )
        return evaluation_config