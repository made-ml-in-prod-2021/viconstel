import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command('train')
@click.option('--data_dir', required=True)
@click.option('--model_dir', required=True)
def train(data_dir: str, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    feature_columns = list(data.columns)
    feature_columns.remove('target')
    X = data[feature_columns]
    y = data[['target']]

    model = LogisticRegression()
    model.fit(X, y)

    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    train()
