FROM airflow-ml-base:latest

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

COPY ./validate.py ./validate.py
ENTRYPOINT [ "python", "./validate.py" ]