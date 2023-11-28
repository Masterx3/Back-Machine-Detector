import os
import sys
import shutil
from GymDetector.logger import logging
from GymDetector.exception import AppException
from GymDetector.entity.config_entity import DataValidationConfig
from GymDetector.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact

class DataValidation:
    def __init__(self, ingestion_artifact: DataIngestionArtifact ,val_config: DataValidationConfig):
        try:
            self.ingestion_artifact = ingestion_artifact
            self.val_config = val_config
        except Exception as e:
            raise AppException(e, sys)
        
    def validate_all_files(self) -> bool:
        try:
            validation_status = True
            
            for file in self.val_config.required_files_list:
                if not os.path.exists(os.path.join(self.ingestion_artifact.feature_store_path, file)):
                    validation_status = False
                
            with open(self.val_config.DATA_VALIDATION_STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                
            return validation_status
        
        except Exception as e:
            raise AppException(e, sys)
        
    def initiate_validation(self) -> DataValidationArtifact:
        logging.info("Data validation stage stated.")
        try:
            status = self.validate_all_files()
            artifact = DataValidationArtifact(val_status=status)
    
            logging.info("validation complete.")
            return artifact
        
        except Exception as e:
            raise AppException(e, sys)