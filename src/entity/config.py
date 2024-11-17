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
    model_data_path : str
    model_output_path : str
    feature_engineered_data : FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    
    def combine_data_fullpath(self):
        self.model_data_path = self.feature_engineered_data.combine_full_path()
        return self.model_data_path



