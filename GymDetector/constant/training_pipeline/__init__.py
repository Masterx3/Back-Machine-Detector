ARTIFACTS_DIR: str = 'artifacts'

# Data ingestion
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'

DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'

DATA_DOWNLOAD_URL: str = 'https://drive.google.com/file/d/1zoLhdlreYb2dMjPwbCiz7KsN5rTTIm_Z/view?usp=sharing'

# Data validation
DATA_VALIDATION_DIR_NAME: str = 'data_validation'

DATA_VALIDATION_STATUS_FILE = 'status.txt'

DATA_VALIDATION_ALL_REQUIRED_FILES = ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'config.yaml']

# Model trainer
MODEL_TRAINER_DIR_NAME: str = 'model_trainer'

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = 'yolov8s.pt'

MODEL_TRAINER_NO_EPOCHS: int = 1000