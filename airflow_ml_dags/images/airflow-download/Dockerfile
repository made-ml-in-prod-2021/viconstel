FROM airflow-ml-base:latest

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./download.py ./download.py
ENTRYPOINT [ "python", "./download.py" ]