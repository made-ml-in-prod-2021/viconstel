FROM airflow-ml-base:latest

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./train.py ./train.py
ENTRYPOINT [ "python", "./train.py" ]