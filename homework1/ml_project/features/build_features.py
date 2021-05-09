from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


class CustomStandardScaler:

    """Custom StandardScaler Class."""

    def __init__(self):
        """Initializer of the `CustomStandardScaler` class."""
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data, y=None):
        """Compute mean and standard deviation over input data."""
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        """Perform standard scaling on input data."""
        data -= self.mean
        data /= self.std
        return data

    def fit_transform(self, data):
        """Perform `fit` and `transform` method after it."""
        self.fit(data)
        return self.transform(data)


class PreprocessingPipeline:

    """Pipeline for data preprocessing."""

    def __init__(self, categorical_features: List[str],
                 numerical_features: List[str]) -> None:
        """Initializer of the `PreprocessingPipeline` class."""
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.pipeline = None

    def get_categorical_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Extract categorical features from DataFrame"""
        return x[self.categorical_features]

    def get_numerical_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features from DataFrame"""
        return x[self.numerical_features]

    def build_categorical_pipeline(self) -> Pipeline:
        """Build pipeline for categorical features."""
        pipeline = Pipeline([
            ('extract_data', FunctionTransformer(self.get_categorical_features)),
            ('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        return pipeline

    def build_numerical_pipeline(self) -> Pipeline:
        """Build pipeline for numerical features."""
        pipeline = Pipeline([
            ('extract_data', FunctionTransformer(self.get_numerical_features)),
            ('impute', SimpleImputer(missing_values=np.nan)),
            ('standard_scaler', CustomStandardScaler())
        ])
        return pipeline

    def fit(self, data: pd.DataFrame) -> None:
        """Fit preprocessing pipeline on the input data."""
        self.pipeline = Pipeline([
            ('pipeline', FeatureUnion([
                ('categorical', self.build_categorical_pipeline()),
                ('numerical', self.build_numerical_pipeline())
            ]))
        ])
        self.pipeline.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing pipeline on the input data."""
        return pd.DataFrame(self.pipeline.transform(data))

    def save_pipeline(self, file_path: str) -> None:
        """Save fitted preprocessing pipeline on the hard drive."""
        joblib.dump(self.pipeline, file_path)

    def load_pipeline(self, file_path: str) -> None:
        """Load fitted preprocessing pipeline from the hard drive."""
        self.pipeline = joblib.load(file_path)
