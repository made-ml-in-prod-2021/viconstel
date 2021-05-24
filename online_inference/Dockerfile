FROM python:3.7
COPY /homework1/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY /homework1/dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install /ml_project-0.1.0.tar.gz

COPY /homework1/models/model.pkl /homework1/models/model.pkl
COPY /homework1/models/preprocessor.pkl /homework1/models/preprocessor.pkl
COPY /homework1/configs/ /homework1/configs
COPY /online_inference/app.py /online_inference/app.py

WORKDIR ./online_inference
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]