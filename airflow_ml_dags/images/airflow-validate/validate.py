import os
import json
import pickle

import click
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


@click.command('validate')
@click.option('--data_dir', required=True)
@click.option('--model_dir', required=True)
def validate(data_dir: str, model_dir: str) -> None:
    data = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    feature_columns = list(data.columns)
    feature_columns.remove('target')
    X_val = data[feature_columns]
    y_val = data[['target']].values

    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_val)

    scores = {
        'f1_score': f1_score(y_val, y_pred),
        'accuracy_score': accuracy_score(y_val, y_pred)
    }

    with open(os.path.join(model_dir, 'metrics.json'), 'w') as file:
        json.dump(scores, file)


if __name__ == '__main__':
    validate()
