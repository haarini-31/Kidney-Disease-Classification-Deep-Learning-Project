import os
import mlflow
from dotenv import load_dotenv
from urllib.parse import urlparse
from pathlib import Path

load_dotenv()

import tensorflow as tf
from tensorflow.keras.models import load_model
import mlflow
from urllib.parse import urlparse
from pathlib import Path

from cnnClassifier.entity.data_ingestion_config import EvaluationConfig
from cnnClassifier.utils.common import save_json, read_yaml, create_directories

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.30)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluate(self):
        self.model = load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        score = {
            "loss": self.score[0],
            "accuracy": self.score[1]
        }
        save_json(path=Path("scores.json"), data=score)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(self.config.mlflow_uri).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("loss", self.score[0])
            mlflow.log_metric("accuracy", self.score[1])

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")