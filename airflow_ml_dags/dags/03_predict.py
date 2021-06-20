import os

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from defaults import DEFAULT_ARGS, RAW_DATA_DIR, PREDICTIONS_DIR


with DAG(
        'predict',
        default_args=DEFAULT_ARGS,
        schedule_interval='@daily',
        start_date=days_ago(0, 2)
) as dag:

    data_sensor = FileSensor(
        filepath=os.path.join(RAW_DATA_DIR, 'data.csv'),
        poke_interval=10,
        task_id='data_sensor'
    )

    model_sensor = FileSensor(
        filepath=Variable.get("model_path"),
        poke_interval=10,
        task_id='model_sensor'
    )

    predict = DockerOperator(
        image='airflow-predict',
        command=f'--input_dir={RAW_DATA_DIR} --model_path={Variable.get("model_path")} --output_dir={PREDICTIONS_DIR}',
        task_id='airflow-predict',
        volumes=[f'{Variable.get("data_dir")}:/data'],
        network_mode='bridge',
        do_xcom_push=False
    )

    [data_sensor, model_sensor] >> predict
