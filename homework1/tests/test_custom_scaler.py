import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from ml_project.features import CustomStandardScaler


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_DATA_PATH = r'data\heart.csv'
TEST_DATA_PATH = r'data\test_sample.csv'


def test_custom_standard_scaler():
    train_df = pd.read_csv(os.path.join(ROOT_DIR, TRAIN_DATA_PATH))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, TEST_DATA_PATH))

    sklearn_scaler = StandardScaler()
    custom_scaler = CustomStandardScaler()

    sklearn_train_data = sklearn_scaler.fit_transform(train_df)
    custom_train_data = custom_scaler.fit_transform(train_df)

    assert np.allclose(sklearn_train_data, custom_train_data.to_numpy()), (
        'Different results with sklearn and custom scaler on train data.'
    )

    sklearn_test_data = sklearn_scaler.fit_transform(test_df)
    custom_test_data = custom_scaler.fit_transform(test_df)

    assert np.allclose(sklearn_test_data, custom_test_data.to_numpy()), (
        'Different results with sklearn and custom scaler on test data.'
    )
