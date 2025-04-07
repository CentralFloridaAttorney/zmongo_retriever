#!/usr/bin/env python3
import logging
from datetime import datetime
from zmongo_toolbag import ZMongo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_model_parameters(zmongo_instance: ZMongo, model_params: dict) -> None:
    """
    Updates (or upserts) the model parameters in MongoDB.

    This function writes the model parameters to a dedicated collection ("model_parameters")
    using a known document identifier (here, "_id": "current_model_parameters").

    Parameters:
      zmongo_instance (ZMongo): An instance of your ZMongo repository.
      model_params (dict): A dictionary of model parameters to save.
    """
    try:
        # Use the synchronous client for simplicity.
        query = {"_id": "current_model_parameters"}
        update = {"$set": model_params}
        result = zmongo_instance.sync_db["model_parameters"].update_one(query, update, upsert=True)
        logger.info(f"Updated model parameters: {result.raw_result}")
    except Exception as e:
        logger.error(f"Error updating model parameters: {e}")


def log_training_metrics(zmongo_instance: ZMongo, metrics: dict) -> None:
    """
    Logs training metrics to the 'training_metrics' collection in MongoDB.

    This function adds a UTC timestamp to the metrics dictionary and then inserts it.

    Parameters:
      zmongo_instance (ZMongo): An instance of your ZMongo repository.
      metrics (dict): A dictionary containing training metrics (e.g. {"loss_D": ..., "loss_G": ...}).
    """
    try:
        metrics_doc = {
            "timestamp": datetime.utcnow(),
            **metrics
        }
        result = zmongo_instance.sync_db["training_metrics"].insert_one(metrics_doc)
        logger.info(f"Logged training metrics: {metrics_doc} (inserted _id: {result.inserted_id})")
    except Exception as e:
        logger.error(f"Error logging training metrics: {e}")


if __name__ == '__main__':
    # Create an instance of ZMongo.
    zmongo_instance = ZMongo()

    # Example: update the model parameters.
    dummy_model_params = {
        "W_G": [[0.1, 0.2], [0.3, 0.4]],  # Example values; your actual model parameters will be larger.
        "b_G": [0.0, 0.0],
        "w_D": [0.5, 0.6],
        "b_D": 0.0
    }
    update_model_parameters(zmongo_instance, dummy_model_params)

    # Example: log some training metrics.
    dummy_metrics = {
        "loss_D": 2.0543,
        "loss_G": 0.6376
    }
    log_training_metrics(zmongo_instance, dummy_metrics)
