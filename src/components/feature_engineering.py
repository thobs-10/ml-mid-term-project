import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from loguru import logger

from src.components.component import Component
from src.entity.config import FeatureEngineeringConfig, DataProcessing, ArtifactStore
from src.utils.main_utils import MixedTypeFeatureSelector
from src.exception  import AppException

class FeatureEngineering(Component):
    def __init__(self, data_source: DataProcessing= DataProcessing(),
                 data_store: FeatureEngineeringConfig= FeatureEngineeringConfig(),
                 artifact_store: ArtifactStore= ArtifactStore()):
        self.data_source = data_source
        self.data_store = data_store
        self.artifact_store = artifact_store
        self.excluded_cols = [
            'Wind speed (m/s)', 
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)', 
            'Snowfall (cm)', 
            'year'
        ]
    
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.data_source.processed_data_path}")
            filepath = os.path.join(self.data_source.processed_data_path)
            file_name = [file for file in os.listdir(filepath) if file.endswith('.csv')]
            df = pd.read_csv(os.path.join(filepath, file_name[0]))
            logger.debug("processed file read sucessfully")
            return df
        except FileNotFoundError as e:
            raise AppException(f"Failed to load data from {self.data_source.processed_data_path}.", e)
    
    def feature_extraction(self, df:pd.DataFrame)-> pd.DataFrame:
        logger.info("feature extraction process ...")
        df['year'] = pd.DatetimeIndex(df['Date']).year
        df['month'] = pd.DatetimeIndex(df['Date']).month
        df['day'] = pd.DatetimeIndex(df['Date']).day
        df['day_name'] = pd.DatetimeIndex(df['Date']).day_name()
        df['is_business_day'] = df['day_name'].apply(lambda x:1 if x not in ['Saturday','Sunday'] else 0)
        logger.debug("feature extraction process done successfully")
        return df
    
    def separate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns='Rented Bike Count', axis=1)
        y = df['Rented Bike Count']
        return X, y
    
    def feature_transformation(self, df:pd.DataFrame)-> pd.DataFrame:
        logger.info("feature transformation process ...")
        df['Holiday'] = np.where(df['Holiday'] =='Holiday', 1, 0)
        df['Seasons'] = df['Seasons'].map({'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4})
        df['Functioning Day'] = np.where(df['Functioning Day'] =='Yes', 1, 0)
        df['day_name'] = df['day_name'].map({'Monday': 1, 'Tuesday': 2, 'Wednsday': 3, 'Thursday': 4, 'Friday': 5})
        df.drop(columns='Date', axis=1, inplace=True)
        logger.debug("feature transformation process done successfully")
        return df
    
    def remove_highly_corrected_features(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("removing highly correlated features process ...")
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        highly_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
        df.drop(columns=highly_corr_features, axis=1, inplace=True)
        logger.debug("removing highly correlated features process done successfully")
        return df
    
    def feature_selection(self, df:pd.DataFrame)-> Tuple[pd.DataFrame, Optional[pd.Series]]:
        logger.info("feature selection process ...")
        X, y = self.separate_dataset(df)
        selector = MixedTypeFeatureSelector(n_features=7)
        X_selected = selector.fit_transform(X, y)
        logger.debug("feature selection process done successfully")
        return X_selected, y

    def split_data(self, X: pd.DataFrame,
               y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                      pd.DataFrame, pd.Series, 
                                      pd.Series, pd.Series]:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2,
                                                                    random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                        test_size=0.25, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
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
    
    def save_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
                  y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,filepath: str) -> None:
        logger.info("saving data ...")
        filepath = os.path.join(self.data_store.feature_engineered_data_path)
        os.makedirs(filepath, exist_ok=True)
        try:
            X_train.to_csv(os.path.join(filepath, "X_train.csv"), index=False)
            X_val.to_csv(os.path.join(filepath, "_X_val.csv"), index=False)
            X_test.to_csv(os.path.join(filepath, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(filepath, "y_train.csv"), index=False)
            y_val.to_csv(os.path.join(filepath, "y_val.csv"), index=False)
            y_test.to_csv(os.path.join(filepath, "y_test.csv"), index=False)
            logger.debug("saving data done successfully")
        except Exception as e:
            logger.error("Error saving data to %s", filepath)
            raise AppException(f"Failed to save data to {filepath}.", e)
        
    def save_artifact(self, scaler: StandardScaler)-> None:
        logger.info("saving artifacts...")
        filepath = os.path.join(self.artifact_store.artifacts_path)
        os.makedirs(filepath, exist_ok=True)
        try:
            joblib.dump(scaler, os.path.join(filepath, "scaler.joblib"))
            logger.debug("saving artifacts done successfully")
        except Exception as e:
            logger.error("Error saving artifacts to %s", filepath)
            raise AppException(f"Failed to save artifacts to {filepath}.", e)