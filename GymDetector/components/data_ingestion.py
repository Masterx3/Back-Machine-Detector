import os
import sys
import zipfile
import gdown
from GymDetector.logger import logging
from GymDetector.exception import AppException
from GymDetector.entity.config_entity import DataIngestionConfig
from GymDetector.entity.artifacts_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = config
            
        except Exception as e:
            return AppException(e, sys)
    
    def download_data(self):
        """
        Downloads the data from URL and unzips it to the artifacts folder
        
        Returns: zip file path
        """
        try:
            url = self.data_ingestion_config.data_download_url
            zip_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_dir, exist_ok=True)
            
            zip_file_path = os.path.join(zip_dir, "data.zip")
            
            if os.path.exists(zip_file_path):
                logging.info("Data.zip already exists. Download skipped.")
                
            else:
                logging.info(f"Downloading data from {url} into file {zip_file_path}")
                
                file_id = url.split('/')[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                gdown.download(prefix+file_id, zip_file_path)
                
                logging.info("Download complete.")
            
            return zip_file_path
        
        except Exception as e:
            return AppException(e, sys)
        
    
    def extract_zip_file(self, zip_path: str) -> str:
        """
        Extracts zip file from given zip_path into specified directory
        
        Returns: Extracted zip path
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_path
            os.makedirs(feature_store_path,exist_ok=True)
        
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_path)
            logging.info(f"Extracted .zip file {zip_path} into {feature_store_path}")
            
            return feature_store_path
        
        except Exception as e:
            return AppException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            file = self.download_data()
            unzipped = self.extract_zip_file(file)
            
            artifcat = DataIngestionArtifact(data_zip_file_path=file, feature_store_path=unzipped)
            
            logging.info('Completed Data Ingestion Stage.')
            logging.info(f"Artifact location: {artifcat}")
            
            return artifcat
        
        except Exception as e:
            raise AppException(e, sys)