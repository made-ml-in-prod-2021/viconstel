import datetime


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["viconstel@gmail.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

RAW_DATA_DIR = 'data/raw/{{ ds }}'
PROCESSED_DATA_DIR = 'data/processed/{{ ds }}'
MODEL_DIR = 'data/models/{{ ds }}'
PREDICTIONS_DIR = 'data/predictions/{{ ds }}'
