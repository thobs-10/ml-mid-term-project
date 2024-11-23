import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def get_statistical_properties(df:pd.DataFrame, column: str) -> Tuple[float, float, float]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR

def get_tree_based_models() -> list:
    """Return a list of tree-based models."""
    return [
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        # CatBoostRegressor(),
        XGBRegressor(),
        LGBMRegressor()
    ]

class MixedTypeFeatureSelector:
    """
    Feature selector for regression problems with mixed categorical and numerical features.
    Combines multiple selection methods:
    1. Mutual Information for non-linear relationships
    2. Spearman Correlation for monotonic relationships
    3. Random Forest importance for complex interactions
    """
    
    def __init__(self, n_features: int=10):
        self.n_features = n_features
        self.feature_scores = None
        self.selected_features = None
        self.excluded_cols = [
            'Wind speed (m/s)', 
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)', 
            'Snowfall (cm)', 
            'year'
        ]
    
    def _clip_values(self,X:pd.DataFrame, col: str) -> None:
        """
        Clip the values to avoid negative scores and zero importance.
        """
        # Clip extreme values
        q1 = X[col].quantile(0.01)
        q3 = X[col].quantile(0.99)
        X[col] = X[col].clip(q1, q3)
    
    def _fill_inf_points(self, X:pd.DataFrame, col:str) -> pd.DataFrame:
        X[col] = X[col].replace(np.inf, X[col].replace([np.inf, -np.inf], np.nan).max())
        X[col] = X[col].replace(-np.inf, X[col].replace([np.inf, -np.inf], np.nan).min())
        X[col] = X[col].fillna(X[col].median())
        return X

    def _scale_data(self,X:pd.DataFrame, numerical_cols: List[str]) -> None:
        cols_to_scale = [col for col in numerical_cols if col not in self.excluded_cols]
        if cols_to_scale:
            scaler = RobustScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector to the data using combination of three(3) different methods for robustness.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Input features (mixed types)
        y : array-like
            Target variable (continuous)
        """

        scores_dict = {}
        X_processed = self._clean_data(X)
        
        # 1. Mutual Information Scores
        mi_scores = mutual_info_regression(X_processed, y)
        scores_dict['mutual_info'] = dict(zip(X.columns, mi_scores))
        
        # 2. Spearman Correlation (absolute values)
        spearman_scores = {}
        for col in X_processed.columns:
            correlation, _ = spearmanr(X_processed[col], y)
            spearman_scores[col] = abs(correlation)
        scores_dict['spearman'] = spearman_scores
        
        # 3. Random Forest Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_processed, y)
        rf_scores = dict(zip(X.columns, rf.feature_importances_))
        scores_dict['random_forest'] = rf_scores
        
        weights = {
            'mutual_info': 0.4,
            'spearman': 0.3,
            'random_forest': 0.3
        }
        final_scores = {}
        for feature in X.columns:
            score = (
                weights['mutual_info'] * self._normalize_score(scores_dict['mutual_info'][feature]) +
                weights['spearman'] * self._normalize_score(scores_dict['spearman'][feature]) +
                weights['random_forest'] * self._normalize_score(scores_dict['random_forest'][feature])
            )
            final_scores[feature] = score

        self.feature_scores = final_scores
        self.selected_features = sorted(final_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:self.n_features]
        
        return self
    
    def _clean_data(self, X:pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        numerical_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col not in self.excluded_cols:
                X_processed[col] = X_processed[col].replace([np.inf, -np.inf], np.nan)
                median_val = X_processed[col].median()
                X_processed[col] = X_processed[col].fillna(median_val)
                
                # Clip extreme values
                # q1 = X_processed[col].quantile(0.01)
                # q3 = X_processed[col].quantile(0.99)
                # X_processed[col] = X_processed[col].clip(q1, q3)
                self._clip_values(X_processed, col)
        
        # cols_to_scale = [col for col in numerical_cols if col not in self.excluded_cols]
        # if cols_to_scale:
        #     scaler = RobustScaler()
        #     X_processed[cols_to_scale] = scaler.fit_transform(X_processed[cols_to_scale])
        self._scale_data(X_processed, numerical_cols)

        for col in self.excluded_cols:
            if col in X_processed.columns:
                # X_processed[col] = X_processed[col].replace(np.inf, X_processed[col].replace([np.inf, -np.inf], np.nan).max())
                # X_processed[col] = X_processed[col].replace(-np.inf, X_processed[col].replace([np.inf, -np.inf], np.nan).min())
                # X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                X_processed = self._fill_inf_points(X_processed, col)
        
        return X_processed
    
    def transform(self, X):
        """Return dataset with only selected features"""
        selected_feature_names = [feature[0] for feature in self.selected_features]
        return X[selected_feature_names]
    
    def fit_transform(self, X, y) -> pd.DataFrame:
        """Fit and transform the data"""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self):
        """Return feature importance scores and ranks"""
        scores_df = pd.DataFrame(self.selected_features, 
                               columns=['Feature', 'Score'])
        scores_df['Rank'] = range(1, len(scores_df) + 1)
        return scores_df
    
    @staticmethod
    def _normalize_score(score):
        """Normalize score to [0, 1] range"""
        return (score - min(score, 0)) / (max(score, 1) - min(score, 0))