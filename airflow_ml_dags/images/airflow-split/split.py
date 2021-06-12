import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command('split')
@click.option('--data_dir', required=True)
def split(data_dir: str) -> None:
    data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    train_data, val_data = train_test_split(data, test_size=0.3)
    train_data.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(data_dir, 'val.csv'), index=False)


if __name__ == '__main__':
    split()
