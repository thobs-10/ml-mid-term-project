import numpy as np
import pandas as pd
import os
import sys
from loguru import logger

from src.components.component import Component
from src.entity.config import DataIngestion, DataProcessing
from src.utils.main_utils import get_statistical_properties
from src.exception  import AppException

class Preprocesing(Component):
    def __init__(self, data_source: DataIngestion = DataIngestion(),
                 data_store: DataProcessing = DataProcessing()):
        self.data_source = data_source
        self.data_store = data_store

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading data...")
            filepath = os.path.join(self.data_source.raw_data_path)
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
            logger.debug(f"Successfully loaded data from {self.data_source.raw_data_path}")
            return df
        except FileNotFoundError as e:
            error_message = f"Failed to load data from {self.data_source.raw_data_path}"
            raise AppException(
                error_message=error_message,
                error_detail=sys
            )
    
    def handle_categorical_types(self, df:pd.DataFrame)-> pd.DataFrame:
        logger.info('Handling categorical datatypes...')
        df['Holiday'] = df['Holiday'].astype('category')
        df['Functioning Day'] = df['Functioning Day'].astype('category')
        df['Seasons'] = df['Seasons'].astype('category')
        logger.debug("Handled categorical datatypes successfully")
        return df
    
    def handle_numeric_types(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info('Handling numeric datatypes...')
        for column in df.columns:
            if pd.api.types.is_object_dtype(df[column]):
                df[column] = df[column].astype(str)
            elif pd.api.types.is_numeric_dtype(df[column]):
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif pd.api.types.is_datetime64_dtype(df[column]):
                df[column] = pd.to_datetime(df[column])
        logger.debug("Handled numeric datatypes successfully")
        return df
    
    def drop_duplicates(self, df:pd.DataFrame)-> pd.DataFrame:
        logger.info("Dropping duplicates...")
        duplicated = df[df.duplicated(keep=False)]
        if duplicated.shape[0] > 0:
            df = df.drop_duplicates(inplace= True)
        logger.debug("Dropped duplicates successfully")
        return df
    
    def handle_missing_values(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values...")
        features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
        for feature in features_with_na:
            if df[feature].dtype == 'categorical':
                df[feature] = df[feature].fillna(df[feature].mode()[0], inplace=True)
            else:
                df[feature] = df[feature].fillna(df[feature].mean())
        logger.debug("Handled missing values successfully")
        return df
    
    def handle_exponential_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling exponential distribution...")
        numerical_cols = df.select_dtypes(include=['number']).columns.to_list()
        numerical_cols.remove('Rented Bike Count')
        for column in numerical_cols:
            Q1, Q3, IQR = get_statistical_properties(df, column)
            outlier = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
            if outlier.shape[0] > 50:
                df[column] = np.log(df[column])
        logger.debug("Handled exponential distribution successfully")
        return df
    
    def save_data(self, df: pd.DataFrame) -> None:
        try:
            logger.info("Saving data...")
            filepath = self.data_store.processed_data_path
            os.makedirs(filepath, exist_ok=True)
            df.to_csv(os.path.join(filepath, "processed_data.csv"), index=False)
            logger.debug("Saved processed data successfully")
        except Exception as e:
            error_message = f"Failed to load data from {self.data_source.raw_data_path}"
            raise AppException(
                error_message=error_message,
                error_detail=sys
            )