import os
from dataclasses import dataclass
from GymDetector.constant.training_pipeline import *

@dataclass(frozen=True)
class TrainingConfig:
    artifacts_dir: str = ARTIFACTS_DIR
    
training_pipeline_config: TrainingConfig = TrainingConfig()

@dataclass(frozen=True)
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME)
    
    feature_store_path: str = os.path.join(training_pipeline_config.artifacts_dir, DATA_INGESTION_FEATURE_STORE_DIR)
    
    data_download_url: str = DATA_DOWNLOAD_URL
    
@dataclass(frozen=True)
class DataValidationConfig:
    DATA_VALIDATION_DIR: str = training_pipeline_config.artifacts_dir

    DATA_VALIDATION_STATUS_FILE = os.path.join(training_pipeline_config.artifacts_dir, DATA_VALIDATION_STATUS_FILE)

    required_files_list = DATA_VALIDATION_ALL_REQUIRED_FILES
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    model_dir = os.path.join(training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME)

    pre_weights = MODEL_TRAINER_PRETRAINED_WEIGHT_NAME
    
    epochs = MODEL_TRAINER_NO_EPOCHS