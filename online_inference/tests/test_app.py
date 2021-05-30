import pytest
from fastapi.testclient import TestClient

from online_inference import app


@pytest.fixture()
def correct_data_sample():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    return [dict.fromkeys(columns, 0), dict.fromkeys(columns, 1)]


@pytest.fixture()
def invalid_data_sample():
    columns = ['sex', 'cp', 'chol', 'fbs', 'restecg', 'slope', 'ca']
    invalid_dict = dict.fromkeys(columns, 0)
    invalid_dict['thal'] = {1: 2}
    return [invalid_dict]


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


def test_app_correct_prediction_request(client, correct_data_sample) -> None:
    response = client.get("/predict", json=correct_data_sample)
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


def test_app_invalid_prediction_request(client, invalid_data_sample) -> None:
    response = client.get("/predict", json=invalid_data_sample)
    assert response.status_code >= 400, (
        'Invalid status code for endpoint `/predict`'
    )
