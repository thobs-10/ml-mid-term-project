from src.exception import AppException

from src.components.feature_engineering import FeatureEngineering
from src.entity.config import FeatureEngineeringConfig, DataProcessing

class FeatureEngineeringPipeline:
    def __init__(self, data_source: DataProcessing = DataProcessing(),
                 data_store: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        self.data_source = data_source
        self.data_store = data_store

    
    def run_feature_engineering(self):
        try:
            feature_engineering = FeatureEngineering(self.data_source,
                                                     self.data_store)
            df = feature_engineering.load_data()
            df = feature_engineering.feature_extraction(df)
            df = feature_engineering.feature_transformation(df)
            df = feature_engineering.remove_highly_corrected_features(df)
            X_selected, y = feature_engineering.feature_selection(df)
            X_train, X_val, X_test, y_train, y_val, y_test = feature_engineering.split_data(X_selected, y)
            # X_train_scaled, X_valid_scaled, scaler = feature_engineering.feature_scaling(X_train, X_val)
            feature_engineering.save_data(X_train, X_val, X_test, y_train, y_val, y_test)
            # feature_engineering.save_artifact(scaler)
        except AppException as e:
            print(f"An error occurred during feature engineering pipeline: {e}")

if __name__ == "__main__":
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    feature_engineering_pipeline.run_feature_engineering()