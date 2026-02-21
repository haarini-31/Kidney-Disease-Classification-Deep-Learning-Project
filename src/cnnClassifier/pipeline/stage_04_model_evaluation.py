from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.data_ingestion_config import EvaluationConfig
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        self.evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=self.evaluation_config)
        evaluation.evaluate()
        evaluation.save_score()
        evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        evaluation_pipeline = EvaluationPipeline()
        evaluation_pipeline.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e