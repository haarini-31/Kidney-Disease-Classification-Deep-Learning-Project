from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")

        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(
            config=prepare_base_model_config
        )

        logger.info("Getting the base model")
        prepare_base_model.get_base_model()

        logger.info("Updating the base model")
        prepare_base_model.update_base_model()

        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")


if __name__ == "__main__":
    try:
        logger.info("==========================================")
        obj=PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info("==========================================")
    except Exception as e:
        logger.exception(e)
        raise e
