import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(dotenv_path=os.path.join(root_dir, '.env'))


@dataclass
class DataIngestion:
    raw_data_path : str = os.getenv('RAW_PATH_FILE')

@dataclass
class DataProcessing:
    processed_data_path: str = os.getenv('PROCESSED_PATH_FILE')

@dataclass
class ArtifactStore:
    artifacts_path: str = os.getenv('ARTIFACTS_PATH')
@dataclass
class FeatureEngineeringConfig:
    feature_engineered_data_path: str =os.getenv('FEATURE_ENGINEERED_DATA_PATH')
    
@dataclass
class ModelTraining:
    model_path : str = os.getenv('MODEL_ARTIFACT_PATH')
    # model_output_path : str = os.getenv('MODEL_OUTPUT_PATH')
    
@dataclass
class ModelConfig:
    hyperparameters : dict = field(default_factory=lambda:{
    # "max_depth": [1, 3, 6, 8, 10],
    "n_estimators": [50, 100, 150, 250, 300], 
    # "loss" : ['squared_error', 'absolute_error', 'huber', 'quantile'],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    # "max_features": ['auto', 'sqrt', 'log2']
})


