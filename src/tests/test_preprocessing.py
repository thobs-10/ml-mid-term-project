import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, Mock
from src.components.preprocessing import Preprocesing
from src.entity.config import DataIngestion, DataProcessing
from src.exception import AppException

@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing"""
    return pd.DataFrame({
        'Holiday': ['Yes', 'No', 'Yes'],
        'Functioning Day': ['Yes', 'No', 'Yes'],
        'Seasons': ['Spring', 'Summer', 'Spring'],
        'Temperature': [20.5, 25.3, 18.7],
        'Humidity': [45, 60, 55],
        'Rented Bike Count': [100, 150, 120],
        'Duplicate_Row': [1, 1, 2]
    })

@pytest.fixture
def preprocessing_instance():
    """Fixture to create a Preprocessing instance"""
    return Preprocesing(
        data_source=DataIngestion(raw_data_path="test_raw_data.csv"),
        data_store=DataProcessing(processed_data_path="test_processed_data")
    )

class TestPreprocessing:
    """Test suite for Preprocessing class"""

    def test_load_data_success(self, preprocessing_instance, sample_df):
        """Test successful data loading"""
        with patch('pandas.read_csv', return_value=sample_df):
            df = preprocessing_instance.load_data()
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

    def test_load_data_file_not_found(self, preprocessing_instance):
        """Test load_data with non-existent file"""
        with patch('pandas.read_csv', side_effect=FileNotFoundError()):
            with pytest.raises(AppException) as exc_info:
                preprocessing_instance.load_data()
            assert "Error occured in python script name" in str(exc_info.value)
            assert "Failed to load data" in str(exc_info.value)

    def test_handle_categorical_types_success(self, preprocessing_instance, sample_df):
        """Test categorical type conversion"""
        processed_df = preprocessing_instance.handle_categorical_types(sample_df)
        
        categorical_columns = ['Holiday', 'Functioning Day', 'Seasons']
        for col in categorical_columns:
            assert pd.api.types.is_categorical_dtype(processed_df[col])

    def test_handle_numeric_types_success(self, preprocessing_instance, sample_df):
        """Test numeric type handling"""
        processed_df = preprocessing_instance.handle_numeric_types(sample_df)
        
        numeric_columns = ['Temperature', 'Humidity', 'Rented Bike Count']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(processed_df[col])
