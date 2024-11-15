import os
from pathlib import Path
from loguru import logger

project_source = 'src'

list_of_files = [
    ".github/workflows/.gitkeep",
    "setup.py",
    "requirements.txt",
    "Dockerfile",
    "Makefile",
    "app.py",
    "LICENSE",
    "README.md",
    ".gitignore",
    ".env",
    "notebooks/.gitkeep",
    F"{project_source}/__init__.py",
    f"{project_source}/components/__init__.py",
    f"{project_source}/components/preprocessing.py",
    f"{project_source}/components/feature_engineering.py",
    f"{project_source}/components/model_trainer.py",
    f"{project_source}/components/model_pusher.py",
    f"{project_source}/components/model_deployer.py",
    f"{project_source}/components/model_evaluation.py",
    f"{project_source}/tests/__init__.py",
    f"{project_source}/utils/__init__.py",
    f"{project_source}/utils/main_utils.py",
    f"{project_source}/entity/__init__.py",
    f"{project_source}/entity/config.py",
    f"{project_source}/exception/__init__.py",
    f"{project_source}/pipelines/__init__.py",
    f"{project_source}/pipelines/data_pipeline.py",
    f"{project_source}/pipelines/feature_eng_pipeline.py",
    f"{project_source}/pipelines/inference_pipeline.py",
    f"{project_source}/run_pipelines.py",
    "data/raw_data/.gitkeep",
    "data/feature_store/.gitkeep",
    "artifacts/.gitkeep"

]

for filepath in list_of_files:
    file_path = Path(filepath)
    file_dir, filename = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logger.info(f"Created directory: {file_dir} for filename {filename}")
    if (not os.path.exists(filename)) or (os.path.getsize()==0):
        with open(file_path, 'w') as file:
            pass
            logger.info(f"Created file: {filename}")
    else:
        logger.info(f"File {filename} already exists.")