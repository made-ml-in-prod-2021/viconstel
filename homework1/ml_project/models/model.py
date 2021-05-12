import pickle
import json
from typing import Union, Dict

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from entities import TrainingParams


def initialize_model(
        params: TrainingParams
) -> Union[LogisticRegression, KNeighborsClassifier]:
    """Initialize `LogisticRegression` or `KNeighborsClassifier` ML model."""
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


def train_model(
        train_data: pd.DataFrame, target: pd.Series, params: TrainingParams
) -> Union[LogisticRegression, KNeighborsClassifier]:
    """Train classifier."""
    model = initialize_model(params)
    model.fit(train_data, target)
    return model


def predict_model(
        model: Union[LogisticRegression, KNeighborsClassifier],
        data: pd.DataFrame) -> np.ndarray:
    """Predict target values for input data."""
    predictions = model.predict(data)
    return predictions


def evaluate_model(predictions: np.ndarray, target: pd.Series) \
        -> Dict[str, float]:
    """Evaluate `accuracy` and `f1_score` metrics."""
    return {
        "accuracy": accuracy_score(target, predictions),
        "f1_score": f1_score(target, predictions)
    }


def save_model(model: Union[LogisticRegression, KNeighborsClassifier],
               file_path: str) -> None:
    """Save trained model on the hard drive."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(
        file_path: str, params: TrainingParams
) -> Union[LogisticRegression, KNeighborsClassifier]:
    """Load pre-trained model from the hard drive."""
    model = initialize_model(params)
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    """Save metrics to the JSON file."""
    with open(file_path, 'w') as file:
        json.dump(metrics, file)
