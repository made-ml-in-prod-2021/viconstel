FROM airflow-ml-base:latest

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./preprocess.py ./preprocess.py
ENTRYPOINT [ "python", "./preprocess.py" ]