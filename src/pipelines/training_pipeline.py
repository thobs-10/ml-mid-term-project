from src.exception import AppException

from src.components.model_trainer import ModelTraining
from src.components.model_tuning import HyperparameterTuning
from src.entity.config import FeatureEngineeringConfig, ArtifactStore

class ModelTrainingPipeline:
    def __init__(self, model_tuning: HyperparameterTuning,
                 data_source:FeatureEngineeringConfig = FeatureEngineeringConfig(),
                 artifact_store: ArtifactStore = ArtifactStore(),
                ):
        self.data_source = data_source
        self.artifact_store = artifact_store
        self.model_tuning = model_tuning

    def run_training_pipeline(self):
        try:
            
            model_trainer = ModelTraining(self.data_source, self.artifact_store)
            X_train, y_train, X_val, y_val = model_trainer.load_data()
            # X_train_scaled, X_valid_scaled, scaler = model_trainer.feature_scaling(X_train,X_val)
            model_trainer.train_model(X_train, y_train)
            model_trainer.validate_model(X_val, y_val)
            trained_model = model_trainer.select_best_model()
            model_trainer.save_data(trained_model)
            # model_trainer.save_artifact(scaler)
            random_cv_model = self.model_tuning.hyperparameter_tuning(X_train, y_train, trained_model)
            best_model = self.model_tuning.cross_validation(random_cv_model, X_train, y_train)
            self.model_tuning.save_pipelined_model(best_model)
            

        except AppException as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    model_tuning = HyperparameterTuning()
    model_training_pipeline = ModelTrainingPipeline(model_tuning)
    model_training_pipeline.run_training_pipeline()