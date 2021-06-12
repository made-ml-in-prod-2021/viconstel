import os
import pickle

import click
import pandas as pd


@click.command('predict')
@click.option('--input_dir', required=True)
@click.option('--model_path', required=True)
@click.option('--output_dir', required=True)
def predict(input_dir: str, model_path: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, 'data.csv'))

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(data)
    predictions_df = pd.DataFrame({'predictions': predictions})

    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)


if __name__ == '__main__':
    predict()
