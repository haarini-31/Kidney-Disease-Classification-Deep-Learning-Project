import os
import sys
import logging

logging_str="%(asctime)s : %(levelname)s : %(name)s : %(message)s"

log_dir='logs'
log_filepath=os.path.join("logs","running_logs.log")
os.makedirs(os.path.dirname(log_filepath),exist_ok=True)

logging.basicConfig(
                    level=logging.INFO,
                    format=logging_str,
                    handlers=[logging.FileHandler(log_filepath),
                              logging.StreamHandler(sys.stdout)]
                    )

logger=logging.getLogger("cnnClassifier")

