from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator

from defaults import DEFAULT_ARGS, RAW_DATA_DIR


with DAG(
        'download',
        default_args=DEFAULT_ARGS,
        schedule_interval='@daily',
        start_date=days_ago(0, 2)
) as dag:

    download = DockerOperator(
        image='airflow-download',
        command=f'--output_dir={RAW_DATA_DIR}',
        task_id='airflow-download',
        volumes=[f'{Variable.get("data_dir")}:/data'],
        network_mode='bridge',
        do_xcom_push=False
    )
