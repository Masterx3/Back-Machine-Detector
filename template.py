import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s:')

project_folder = 'GymDetector'

list_of_files = [
    ".github/workflows/.gitkeep",
    "data/.gitkeep",
    f"{project_folder}/__init__.py",
    f"{project_folder}/components/__init__.py",
    f"{project_folder}/components/data_ingestion.py",
    f"{project_folder}/components/data_validation.py",
    f"{project_folder}/components/model_trainer.py",
    f"{project_folder}/constant/__init__.py",
    f"{project_folder}/constant/training_pipeline/__init__.py",
    f"{project_folder}/constant/application.py",
    f"{project_folder}/entity/config_entity.py",
    f"{project_folder}/entity/artifacts_entity.py",
    f"{project_folder}/exception/__init__.py",
    f"{project_folder}/logger/__init__.py",
    f"{project_folder}/pipeline/__init__.py",
    f"{project_folder}/pipeline/training_pipeline.py",
    f"{project_folder}/utils/__init__.py",
    f"{project_folder}/utils/main_utils.py",
    "research/trials.ipynb",
    "templates/index.html",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
]

# Creating logic for creating the files
for string in list_of_files:
    filepath = Path(string)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        if os.path.exists(filedir):
            logging.info(f"The directory {filedir} already exists.")
        else:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating empty folder: {filedir}")

            
    if os.path.exists(filepath):
        logging.info(f"File: {filepath} already present.")
        
    else:
        logging.info(f"Creating empty file: {filepath}")
        with open(filepath, 'w') as f:
            pass