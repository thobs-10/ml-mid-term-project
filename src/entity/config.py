from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class DataIngestion(BaseModel):
    raw_data_path : str

@dataclass
class DataProcessing:
    processed_data_path: str 

@dataclass
class FeatureEngineering:
    data_processing: DataProcessing = field(default_factory=DataProcessing)
    engineered_data_path: str
    
    def combine_full_path(self):
        self.engineered_data_path = self.data_processing.processed_data_path
        return self.engineered_data_path

@dataclass
class ModelTraining:
    model_data_path : str
    model_output_path : str
    feature_engineered_data : FeatureEngineering = field(default_factory=FeatureEngineering)
    
    def combine_data_fullpath(self):
        self.model_data_path = self.feature_engineered_data.combine_full_path()
        return self.model_data_path



