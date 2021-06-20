import sys

import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder='dags', include_examples=False)


@pytest.fixture()
def download_dag_structure():
    structure = {
        'airflow-download': []
    }
    return structure


@pytest.fixture()
def train_dag_structure():
    structure = {
        'train_data_sensor': ['airflow-preprocess'],
        'target_data_sensor': ['airflow-preprocess'],
        'airflow-preprocess': ['airflow-split'],
        'airflow-split': ['airflow-train'],
        'airflow-train': ['airflow-validate'],
        'airflow-validate': []
    }
    return structure


@pytest.fixture()
def predict_dag_structure():
    structure = {
        'data_sensor': ['airflow-predict'],
        'model_sensor': ['airflow-predict'],
        'airflow-predict': []
    }
    return structure


def test_load_dags(dag_bag):
    assert dag_bag.import_errors == {}
    assert dag_bag.dags is not None
    assert set(dag_bag.dags) == {'download', 'train', 'predict'}


def test_download_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id='download')
    assert dag is not None
    assert len(dag.tasks) == 1


def test_train_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id='train')
    assert dag is not None
    assert len(dag.tasks) == 6


def test_predict_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id='predict')
    assert dag is not None
    assert len(dag.tasks) == 3


def test_download_dag_structure(dag_bag, download_dag_structure):
    dag = dag_bag.get_dag(dag_id='download')
    assert dag.task_dict.keys() == download_dag_structure.keys()
    for task_id, downstream_list in download_dag_structure.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_train_dag_structure(dag_bag, train_dag_structure):
    dag = dag_bag.get_dag(dag_id='train')
    assert dag.task_dict.keys() == train_dag_structure.keys()
    for task_id, downstream_list in train_dag_structure.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_predict_dag_structure(dag_bag, predict_dag_structure):
    dag = dag_bag.get_dag(dag_id='predict')
    assert dag.task_dict.keys() == predict_dag_structure.keys()
    for task_id, downstream_list in predict_dag_structure.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)
