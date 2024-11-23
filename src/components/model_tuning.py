import joblib
import os
import pandas as pd
import numpy as np
from typing import Union
from sklearn.model_selection import  RandomizedSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from src.exception import AppException
from src.entity.config import ModelConfig, FeatureEngineeringConfig, ArtifactStore
from loguru import logger

class HyperparameterTuning:
    def __init__(self, hyperparameters: ModelConfig = ModelConfig(), 
                 data_source: FeatureEngineeringConfig = FeatureEngineeringConfig(),
                 model_source: ArtifactStore = ArtifactStore()):
        self.hyperparameters = hyperparameters.hyperparameters
        self.data_source = data_source
        self.model_source = model_source
    
    def load_model(self):
        """load model from artifacts"""
        filepath = os.path.join(self.model_source.artifacts_path, 'trained_model')
        model_path = os.path.join(filepath,"trained_model.joblib")
        try:
            model = joblib.load(model_path)
            return model
        except FileNotFoundError as e:
            raise AppException("Failed to load model", e)
    
    
    def hyperparameter_tuning(self, X_train_scaled:np.ndarray, y_train:pd.DataFrame, model) -> RandomizedSearchCV:
        """hyperparameter tuning"""
        logger.info("hyperparameter tuning...")
        random_cv_model = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=self.hyperparameters,
                                n_iter=3,
                                cv=5,
                                verbose=2,
                                n_jobs=-1)
        random_cv_model.fit(X_train_scaled, y_train)
        logger.debug("hyperparameter tuning succeeded")
        return random_cv_model

    def cross_validation(self,random_cv_model: RandomizedSearchCV, X_train_scaled: np.ndarray, y_train: pd.DataFrame)-> Union[BaseEstimator]:
        logger.info("cross_validation ...")
        kf = KFold(n_splits=5)
        best_model = random_cv_model.best_estimator_
        scores = cross_val_score(best_model, X_train_scaled, y_train, cv=kf)
        if scores.mean() > 0.80:
            # self.save_model(best_model)
            logger.info("Model passed the best score")
            logger.debug("cross validation successful")
            return best_model
        return None
    
    def load_scaler(self)-> StandardScaler:
        scaler_filepath = os.path.join(self.model_source.artifacts_path, 'scaler')
        try:
            logger.info("Loading scaler ...")
            scaler = joblib.load(scaler_filepath)
            logger.debug("Scaler loaded successfully")
            return scaler
        except FileNotFoundError as e:
            raise AppException("Failed to load scaler", e)
        
    def save_pipelined_model(self,model: BaseEstimator):
        # scaler = self.load_scaler()
        model_pipeline  = make_pipeline( model)
        staged_model_path = os.path.join(self.model_source.artifacts_path, 'staged_model')
        os.makedirs(staged_model_path, exist_ok= True)
        try:
            joblib.dump(model_pipeline, os.path.join(staged_model_path, 'model_pipeline.joblib'))
            logger.debug("Pipelined model saved successfully")
        except Exception as e:
            raise AppException("Failed to save pipelined model", e)
    
    
