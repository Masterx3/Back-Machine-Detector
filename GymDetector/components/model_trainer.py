import os
import sys
import comet_ml
from pathlib import Path
# import torch
from ultralytics import YOLO
from GymDetector.logger import logging
from GymDetector.exception import AppException
from GymDetector.utils.main_utils import read_yaml
from GymDetector.constant.training_pipeline import *
from GymDetector.entity.config_entity import ModelTrainerConfig
from GymDetector.entity.artifacts_entity import ModelTrainingArtifact

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def initiate_model_training(self) -> ModelTrainingArtifact:
        logging.info("Model traininig stage started")
        os.makedirs(self.config.model_dir, exist_ok= True)
        comet_ml.init(api_key = "gdaK0f55Z5bQ9lF5BNAwBPG96") # using cometml for experiment tracking
        
        try:
            os.system(f"yolo task=detect mode=train model={self.config.pre_weights} data=config.yaml epochs={self.config.epochs} imgsz=640 save=true")
            os.system(f"cp -r runs/ {self.config.model_dir}/")
            # os.system("rm -rf runs")
            
            model_trainer_artifact = ModelTrainingArtifact(trained_model_file_path = os.path.join(self.config.model_dir, "runs/detect/train", "best.pt"))
            
            logging.info(f"Finished training stage. Model and Results are in {self.config.model_dir}")
            
            return model_trainer_artifact
            
        except Exception as e:
            return AppException(e, sys)