import os
from sklearn.metrics import r2_score
from typing import Tuple, Union
import numpy as np
import pandas as pd
import joblib
from src.exception import AppException
from loguru import logger
from src.components.component import Component
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator
from src.utils.main_utils import get_tree_based_models
from src.entity.config import FeatureEngineeringConfig, ArtifactStore


class ModelTraining(Component):
    def __init__(self, data_source : FeatureEngineeringConfig = FeatureEngineeringConfig(),
                 artifact_data_store : ArtifactStore = ArtifactStore()):
        self.data_source = data_source
        self.artifact_data_store = artifact_data_store
        self.models = []
        self.accuracy_dict = {}
        self.excluded_cols = [
            'Wind speed (m/s)', 
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)', 
            'Snowfall (cm)',
            'year',
            'Seasons']
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """load data for training"""
        logger.info("Loading data for training...")
        try:
            filepath = os.path.join(self.data_source.feature_engineered_data_path)
            X_train = pd.read_csv(os.path.join(filepath, 'X_train.csv'))
            y_train = pd.read_csv(os.path.join(filepath, 'y_train.csv'))
            X_val = pd.read_csv(os.path.join(filepath, '_X_val.csv'))
            y_val = pd.read_csv(os.path.join(filepath, 'y_val.csv'))
            logger.debug("Training data sucessfully loaded")
            return X_train, y_train, X_val, y_val
        except FileNotFoundError as e:
            raise AppException("Failed to load data for training.", e)
       
    
    def _clean_infinity_values(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.excluded_cols:
            if col in X.columns:
                X[col] = X[col].replace(np.inf, X[col].replace([np.inf, -np.inf], np.nan).max())
                X[col] = X[col].replace(-np.inf, X[col].replace([np.inf, -np.inf], np.nan).min())
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def feature_scaling(self, X_train:pd.DataFrame,
                    X_valid: pd.DataFrame)-> Tuple[Union[pd.DataFrame, np.ndarray],
                                                   Union[pd.DataFrame, np.ndarray],
                                                   StandardScaler]:
        logger.info("feature scaling process ...")
        scaler = MinMaxScaler()
        X_train = self._clean_infinity_values(X_train)
        scaling_cols = [col for col in X_train.columns if col not in self.excluded_cols]
        X_train_scaled = scaler.fit_transform(X_train[scaling_cols])
        X_valid_scaled = scaler.transform(X_valid[scaling_cols])
        logger.debug("feature scaling process done successfully")
        return X_train_scaled, X_valid_scaled, scaler

    def train_model(self, X_train_scaled:pd.DataFrame, y_train:pd.DataFrame) -> None:
        """train tree based models only"""
        logger.info("training models ...")
        X_train_scaled = self._clean_infinity_values(X_train_scaled)
        tree_models  = get_tree_based_models()
        for idx, model in enumerate(tree_models):
            model.fit(X_train_scaled, y_train)
            self.models.append(model)
        logger.debug("training process done successfully")

    def validate_model(self, X_val_scaled: pd.DataFrame, y_val: pd.DataFrame) -> None:
        """validate tree based models only"""
        logger.info("validating models ...")
        X_val_scaled = self._clean_infinity_values(X_val_scaled)
        for model in self.models:
            y_pred = model.predict(X_val_scaled)
            self.accuracy_dict[model.__class__.__name__] = r2_score(y_val, y_pred)
            logger.info(f"Accuracy of {model.__class__.__name__} model: {r2_score(y_val, y_pred)}")
        logger.debug("validation process done successfully")

    def select_best_model(self)-> BaseEstimator:
        """select best tree based model only"""
        logger.info("selecting the best model..")
        best_model = max(zip(self.accuracy_dict.values(), self.accuracy_dict.keys()))[1]
        logger.info(f"Best model: {best_model}")
        # for m in self.models:
        #     if best_model == m.__class__.__name__:
        #         wanted_model = m
        wanted_model = [m for m in self.models if m.__class__.__name__ == best_model][0]
        logger.debug("best model selection done successfully")
        return wanted_model
    
    def save_data(self, model) -> None:
        """save best tree based model only"""
        filepath = os.path.join(self.artifact_data_store.artifacts_path, 'trained_model')
        os.makedirs(filepath, exist_ok=True)
        try:
            logger.info("saving model...")
            joblib.dump(model, os.path.join(filepath, "trained_model.joblib"))
            logger.debug("saving model done successfully")
        except Exception as e:
            raise AppException("Failed to save model", e)
        
    def save_artifact(self, scaler: MinMaxScaler)-> None:
        logger.info("saving artifacts...")
        filepath = os.path.join(self.artifact_data_store.artifacts_path, 'scaler')
        os.makedirs(filepath, exist_ok=True)
        try:
            joblib.dump(scaler, os.path.join(filepath, "scaler.joblib"))
            logger.debug("saving artifacts done successfully")
        except Exception as e:
            logger.error("Error saving artifacts to %s", filepath)
            raise AppException(f"Failed to save artifacts to {filepath}.", e)