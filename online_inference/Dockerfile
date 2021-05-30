FROM python:3.7-slim-stretch
COPY /homework1/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY /homework1/dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install --no-cache-dir /ml_project-0.1.0.tar.gz

COPY /homework1/models/model.pkl /homework1/models/model.pkl
COPY /homework1/models/preprocessor.pkl /homework1/models/preprocessor.pkl
COPY /online_inference/app.py /online_inference/app.py

ENV PATH_TO_MODEL='../homework1/models/model.pkl'
ENV PATH_TO_PREPROCESSOR='../homework1/models/preprocessor.pkl'

WORKDIR ./online_inference
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]