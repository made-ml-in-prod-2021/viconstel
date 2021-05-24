import pytest
from fastapi.testclient import TestClient

from online_inference import app

CORRECT_DATA_SAMPLE = [
    {
        'age': 0,
        'sex': 0,
        'cp': 0,
        'trestbps': 0,
        'chol': 0,
        'fbs': 0,
        'restecg': 0,
        'thalach': 0,
        'exang': 0,
        'oldpeak': 0,
        'slope': 0,
        'ca': 0,
        'thal': 0,
    },
    {
        'age': 1,
        'sex': 1,
        'cp': 1,
        'trestbps': 1,
        'chol': 1,
        'fbs': 1,
        'restecg': 1,
        'thalach': 1,
        'exang': 1,
        'oldpeak': 1,
        'slope': 1,
        'ca': 1,
        'thal': 1,
    }
]
INVALID_DATA_SAMPLE = [
  {
    'sex': 0,
    'cp': 0,
    'chol': 0,
    'fbs': 0,
    'restecg': 0,
    'slope': 0,
    'ca': 0,
    'thal': {1: 2},
  }
]


@pytest.fixture()
def client():
    with TestClient(app) as rest_client:
        yield rest_client


def test_app_root_endpoint(client) -> None:
    response = client.get("/")
    assert response.status_code == 200, (
        'Invalid status code for endpoint `/`'
    )


def test_app_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200, (
        'Invalid status code for endpoint `/health`'
    )
    assert response.json() is True, (
        'Model or preprocessor is None'
    )


def test_app_docs_endpoint(client) -> None:
    response = client.get("/docs")
    assert response.status_code == 200, (
        'Invalid status code for endpoint `/docs`'
    )


def test_app_invalid_endpoint(client) -> None:
    response = client.get("/invalid_endpoint")
    assert response.status_code >= 400, (
        'Invalid status code for unknown endpoint'
    )


def test_app_correct_prediction_request(client) -> None:
    response = client.get("/predict", json=CORRECT_DATA_SAMPLE)
    assert response.status_code == 200, (
        'Invalid status code for endpoint `/predict`'
    )

    prediction_item = response.json()[0]
    item_id = int(prediction_item['id'])
    prediction = int(prediction_item['prediction'])
    assert item_id == 0, (
        f'Wrong id: {item_id}'
    )
    assert prediction == 0 or prediction == 1, (
        'Wrong prediction result'
    )

    prediction_item = response.json()[1]
    item_id = int(prediction_item['id'])
    prediction = int(prediction_item['prediction'])
    assert item_id == 1, (
        f'Wrong id: {item_id}'
    )
    assert prediction == 0 or prediction == 1, (
        'Wrong prediction result'
    )


def test_app_invalid_prediction_request(client) -> None:
    response = client.get("/predict", json=INVALID_DATA_SAMPLE)
    assert response.status_code >= 400, (
        'Invalid status code for endpoint `/predict`'
    )
