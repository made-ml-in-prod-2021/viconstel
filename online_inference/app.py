import os
from typing import List, Optional, Union

import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ml_project.entities import read_pipeline_params
from ml_project.features import PreprocessingPipeline
from ml_project.models import predict_model, load_model


ROOT_DIR = '../homework1'
CONFIG_PATH = r'configs/logreg_config.yml'


model: Optional[Union[LogisticRegression, KNeighborsClassifier]] = None
preprocessor: Optional[Pipeline] = None


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


def make_dataframe(data: List[InputData]) -> pd.DataFrame:
    """Build pd.Dataframe from input data."""
    df = pd.DataFrame(columns=list(InputData.__fields__.keys()))
    for item in data:
        df = df.append(dict(item), ignore_index=True)
    return df


def make_prediction(data: pd.DataFrame) -> List[RequestResponse]:
    """Make prediction for input data and build REST response."""
    processed_data = preprocessor.transform(data)
    predictions = predict_model(model, processed_data)
    response = []
    for i in data.index:
        response.append(RequestResponse(id=i, prediction=predictions[i]))
    return response


app = FastAPI(title='Heart Disease Classifier')


@app.on_event("startup")
def load_model_and_preprocessor() -> None:
    """Initial loading of classifier model and data preprocessor."""
    global model, preprocessor
    params = read_pipeline_params(os.path.join(ROOT_DIR, CONFIG_PATH))
    model_path = os.path.join(ROOT_DIR, params.output_model_path)
    model = load_model(model_path)
    preproc_path = os.path.join(ROOT_DIR, params.output_preprocessor_path)
    preprocessor = PreprocessingPipeline(
        params.feature_params.categorical_features,
        params.feature_params.numerical_features
    )
    preprocessor.load_pipeline(preproc_path)


@app.get("/")
def main() -> str:
    return "It is entry point of our predictor"


@app.get(
    "/health",
    description='Check model and preprocessor health'
)
def health() -> bool:
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
