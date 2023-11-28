import os
import sys
from GymDetector.logger import logging
from GymDetector.exception import AppException
from GymDetector.components.data_ingestion import DataIngestion
from GymDetector.components.data_validation import DataValidation
from GymDetector.components.model_trainer import ModelTrainer
from GymDetector.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig
from GymDetector.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainingArtifact

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_val_config = DataValidationConfig()
        self.train_config = ModelTrainerConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
            try:
                logging.info(
                "Entered data ingestion stage."
            )
            
                prepare_data_ingestion = DataIngestion(config = self.data_ingestion_config)
                data_ingestion_artifcat = prepare_data_ingestion.initiate_data_ingestion()
                logging.info("Data Ingestion complete.")
                return data_ingestion_artifcat
            
            except Exception as e:
                return AppException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Entered data validation stage.")
        
        try:
            prepare_data_val = DataValidation(ingestion_artifact= data_ingestion_artifact, val_config=self.data_val_config)
            data_val_artifact = prepare_data_val.initiate_validation()
            
            logging.info("Data validation complete.")
            
            return data_val_artifact
        
        except Exception as e:
            raise AppException(e, sys)
        
    def start_model_trainer(self, train_config: ModelTrainerConfig) -> ModelTrainingArtifact:
        try:
            model_trainer = ModelTrainer(config=train_config)
            model_trainer_artifact = model_trainer.initiate_model_training()
            return model_trainer_artifact
        
        except Exception as e:
            return AppException(e, sys)
        
    def run_pipeline(self) -> None:
        try:
            data_artifact = self.start_data_ingestion()
            val_artifact = self.start_data_validation(data_ingestion_artifact= data_artifact)
            
            if val_artifact.val_status:
                model_trainer_artifact = self.start_model_trainer(self.train_config)
            
            else:
                raise Exception(
                    "Validation failed. Your data has files missing. Check DATA_VALIDATION_ALL_REQUIRED_FILES in __init__.py of training pipeline folder")
                
        except Exception as e:
            raise AppException(e, sys)