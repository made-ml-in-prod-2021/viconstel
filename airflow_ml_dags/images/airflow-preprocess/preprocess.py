import os

import click
import pandas as pd


@click.command('preprocess')
@click.option('--input_dir', required=True)
@click.option('--output_dir', required=True)
def preprocess(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    target = pd.read_csv(os.path.join(input_dir, 'target.csv'))
    os.makedirs(output_dir, exist_ok=True)
    processed_data = pd.concat([data, target], axis=1)
    output_path = os.path.join(output_dir, 'train_data.csv')
    processed_data.to_csv(output_path, index=False)


if __name__ == '__main__':
    preprocess()
