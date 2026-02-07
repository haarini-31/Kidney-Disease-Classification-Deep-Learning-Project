from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from cnnClassifier.components.model_training import Training # Ensure this path is correct
from pathlib import Path


STAGE_NAME = "Training Stage"

def main():
    # 1. Initialize Configuration
    config = ConfigurationManager()
    
    # 2. Get the Training Configuration (Check for typos here!)
    training_config = config.get_training_config()
    
    # 3. Initialize Training Component
    training = Training(config=training_config)
    
    # 4. Execute Training Pipeline
    training.get_base_model()
    training.train_valid_generator()
    training.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e