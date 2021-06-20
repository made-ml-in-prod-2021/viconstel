import os
import time
import joblib
from typing import List
from datetime import datetime

import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI


PATH_TO_MODEL = '../homework1/models/model.pkl'
PATH_TO_PREPROCESSOR = '../homework1/models/preprocessor.pkl'
START_DELAY = 30
WORK_TIME = 60


model = None
preprocessor = None


class RequestResponse(BaseModel):
    id: int
    prediction: float


class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


def read_pickle(filepath: str):
    obj = joblib.load(filepath)
    return obj


def make_dataframe(data: List[InputData]) -> pd.DataFrame:
    """Build pd.Dataframe from input data."""
    df = pd.DataFrame(columns=list(InputData.__fields__.keys()))
    for item in data:
        df = df.append(dict(item), ignore_index=True)
    return df


def make_prediction(data: pd.DataFrame) -> List[RequestResponse]:
    """Make prediction for input data and build REST response."""
    processed_data = preprocessor.transform(data)
    predictions = model.predict_model(processed_data)
    response = []
    for i in data.index:
        response.append(RequestResponse(id=i, prediction=predictions[i]))
    return response


start_time = datetime.now()
app = FastAPI(title='Heart Disease Classifier')


@app.on_event("startup")
def load_model_and_preprocessor() -> None:
    """Initial loading of classifier model and data preprocessor."""
    global model, preprocessor
    time.sleep(START_DELAY)
    model_path = os.getenv("PATH_TO_MODEL") \
        if os.getenv("PATH_TO_MODEL") else PATH_TO_MODEL
    model = read_pickle(model_path)
    preproc_path = os.getenv("PATH_TO_PREPROCESSOR") \
        if os.getenv("PATH_TO_PREPROCESSOR") else PATH_TO_PREPROCESSOR
    preprocessor = read_pickle(preproc_path)


@app.get("/")
def main() -> str:
    return "It is entry point of our predictor"


@app.get(
    "/health",
    description='Check model and preprocessor health'
)
def health() -> bool:
    if (datetime.now() - start_time).seconds > WORK_TIME:
        raise OSError('Application stop')
    return not (model is None and preprocessor is None)


@app.get(
    "/predict",
    response_model=List[RequestResponse],
    description='Make model prediction for input data'
)
def predict(request: List[InputData]) -> List[RequestResponse]:
    request_df = make_dataframe(request)
    return make_prediction(request_df)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
