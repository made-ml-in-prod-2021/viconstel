FROM airflow-ml-base:latest

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./predict.py ./predict.py
ENTRYPOINT [ "python", "./predict.py" ]