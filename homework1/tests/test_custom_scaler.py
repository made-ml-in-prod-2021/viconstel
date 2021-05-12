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


def test_custom_standard_scaler_zero_variance():
    sklearn_scaler = StandardScaler()
    custom_scaler = CustomStandardScaler()

    multi_column_df = pd.DataFrame({'col1': np.ones(7), 'col2': 5 * np.ones(7)})
    sklearn_data = sklearn_scaler.fit_transform(multi_column_df)
    custom_data = custom_scaler.fit_transform(multi_column_df)

    assert np.allclose(sklearn_data, custom_data.to_numpy()), (
        'Dataframe with multiple constant columns processed incorrectly.'
    )

    single_column_df = pd.DataFrame({'col1': np.ones(10)})
    sklearn_data = sklearn_scaler.fit_transform(single_column_df)
    custom_data = custom_scaler.fit_transform(single_column_df)

    assert np.allclose(sklearn_data, custom_data.to_numpy()), (
        'Dataframe with single constant columns processed incorrectly.'
    )

    mixed_column_df = pd.DataFrame({'col1': np.ones(10), 'col2': np.arange(10)})
    sklearn_data = sklearn_scaler.fit_transform(mixed_column_df)
    custom_data = custom_scaler.fit_transform(mixed_column_df)

    assert np.allclose(sklearn_data, custom_data.to_numpy()), (
        'Dataframe with constant and non-constant columns processed incorrectly.'
    )
