from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities import SplitParams


def read_data(file_path: str) -> pd.DataFrame:
    """Read CSV file"""
    data = pd.read_csv(file_path)
    return data


def split_train_val_data(data: pd.DataFrame, params: SplitParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data on train and validation subsets"""
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
