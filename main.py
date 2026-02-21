import dagshub
dagshub.init(
    repo_owner="haarini-31",
    repo_name="Kidney-Disease-Classification-Deep-Learning-Project",
    mlflow=True
)
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    training = ModelTrainingPipeline()
    training.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n")  
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
    evaluation = EvaluationPipeline()
    evaluation.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

