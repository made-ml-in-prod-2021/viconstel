import joblib
import json
from typing import Union, Dict

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ml_project.entities import TrainingParams


class CustomModel:
    def __init__(self, params: TrainingParams):
        self.model = self._initialize_model(params)

    def _initialize_model(self, params: TrainingParams) \
            -> Union[LogisticRegression, KNeighborsClassifier]:
        """Initialize `LogisticRegression` or `KNeighborsClassifier` model."""
        if params.model_type == 'LogisticRegression':
            model = LogisticRegression(
                penalty=params.penalty,
                C=params.inverse_regularization_strength,
                fit_intercept=params.fit_intercept,
                solver=params.solver,
                max_iter=params.max_iter,
                random_state=params.random_state
            )
        elif params.model_type == 'KNeighborsClassifier':
            model = KNeighborsClassifier(
                n_neighbors=params.n_neighbors,
                algorithm=params.algorithm,
                metric=params.metric
            )
        else:
            raise NotImplementedError
        return model

    def train_model(self, train_data: pd.DataFrame, target: pd.Series) -> None:
        """Train classifier."""
        self.model.fit(train_data, target)

    def predict_model(self, data: pd.DataFrame) -> np.ndarray:
        """Predict target values for input data."""
        predictions = self.model.predict(data)
        return predictions


def evaluate_model(predictions: np.ndarray, target: pd.Series) \
        -> Dict[str, float]:
    """Evaluate `accuracy` and `f1_score` metrics."""
    return {
        "accuracy": accuracy_score(target, predictions),
        "f1_score": f1_score(target, predictions)
    }


def save_model(model: CustomModel, file_path: str) -> None:
    """Save trained model on the hard drive."""
    with open(file_path, 'wb') as file:
        joblib.dump(model, file)


def load_model(file_path: str) -> CustomModel:
    """Load pre-trained model from the hard drive."""
    with open(file_path, 'rb') as file:
        model = joblib.load(file)
    return model


def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    """Save metrics to the JSON file."""
    with open(file_path, 'w') as file:
        json.dump(metrics, file)
