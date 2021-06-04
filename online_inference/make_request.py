import os

import pandas as pd
import requests


DIR_PATH = '../homework1'
FILE_PATH = 'data/heart.csv'


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DIR_PATH, FILE_PATH))
    df_sample = df.sample(50)
    for i in range(len(df_sample)):
        data = df_sample.iloc[i].to_dict()
        print(data)
        response = requests.get('http://0.0.0.0:8000/predict', json=[data])
        print(response.status_code)
        print(response.json())
        print()
