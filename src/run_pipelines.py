
from src.components.model_tuning import HyperparameterTuning
from src.pipelines.data_pipeline import PreprocessingPipeline
from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline
from src.pipelines.training_pipeline import ModelTrainingPipeline

def run_data_pipeline():
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.run_data_processing()


def run_feature_engineering_pipeline():
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    feature_engineering_pipeline.run_feature_engineering()

def run_model_training_pipeline():
    model_tuning = HyperparameterTuning()
    model_training_pipeline = ModelTrainingPipeline(model_tuning)
    model_training_pipeline.run_training_pipeline()


if __name__ == "__main__":
    run_data_pipeline()
    run_feature_engineering_pipeline()
    run_model_training_pipeline()