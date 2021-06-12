import os

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from defaults import DEFAULT_ARGS, PROCESSED_DATA_DIR, RAW_DATA_DIR, MODEL_DIR


with DAG(
        'train',
        default_args=DEFAULT_ARGS,
        schedule_interval='@weekly',
        start_date=days_ago(0, 2)
) as dag:
    train_sensor = FileSensor(
        filepath=os.path.join(RAW_DATA_DIR, 'data.csv'),
        poke_interval=10,
        task_id='train_data_sensor'
    )

    target_sensor = FileSensor(
        filepath=os.path.join(RAW_DATA_DIR, 'target.csv'),
        poke_interval=10,
        task_id='target_data_sensor'
    )

    preprocess = DockerOperator(
        image='airflow-preprocess',
        command=f'--input_dir={RAW_DATA_DIR} --output_dir={PROCESSED_DATA_DIR}',
        task_id='airflow-preprocess',
        volumes=[f'{Variable.get("data_dir")}:/data'],
        network_mode='bridge',
        do_xcom_push=False
    )

    split = DockerOperator(
        image='airflow-split',
        command=f'--data_dir={PROCESSED_DATA_DIR}',
        task_id='airflow-split',
        volumes=[f'{Variable.get("data_dir")}:/data'],
        network_mode='bridge',
        do_xcom_push=False
    )

    train = DockerOperator(
        image='airflow-train',
        command=f'--data_dir={PROCESSED_DATA_DIR} --model_dir={MODEL_DIR}',
        task_id='airflow-train',
        volumes=[f'{Variable.get("data_dir")}:/data'],
        network_mode='bridge',
        do_xcom_push=False
    )

    validate = DockerOperator(
        image='airflow-validate',
        command=f'--data_dir={PROCESSED_DATA_DIR} --model_dir={MODEL_DIR}',
        task_id='airflow-validate',
        volumes=[f'{Variable.get("data_dir")}:/data'],
        network_mode='bridge',
        do_xcom_push=False
    )

    [train_sensor, target_sensor] >> preprocess >> split >> train >> validate
