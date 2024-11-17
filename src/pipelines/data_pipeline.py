from src.exception import AppException

from src.components.preprocessing import Preprocesing
from src.entity.config import DataIngestion, DataProcessing

class PreprocessingPipeline:
    def __init__(self, data_source_path: DataIngestion= DataIngestion(), 
                 data_store_path:DataProcessing = DataProcessing):
        self.data_source_path = data_source_path
        self.data_store_path = data_store_path

    def run_data_processing(self):
        try:
            preprocessing = Preprocesing(self.data_source_path, self.data_store_path)
            df = preprocessing.load_data()
            df = preprocessing.handle_categorical_types(df)
            df = preprocessing.handle_numeric_types(df)
            df = preprocessing.drop_duplicates(df)
            df = preprocessing.handle_missing_values(df)
            df = preprocessing.handle_exponential_distribution(df)
            preprocessing.save_data(df)
        except AppException as e:
            print(f"An error occurred during data preprocessing pipeline: {e}")


if __name__ == "__main__":
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.run_data_processing()