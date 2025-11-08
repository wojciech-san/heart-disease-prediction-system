FROM python:3.12-slim


RUN pip install pipenv


WORKDIR /app


COPY ["Pipfile", "Pipfile.lock", "./"]


RUN pipenv install --system --deploy

COPY ["predict.py", "./"]


COPY ["models/random_forest_heart_disease_v1.bin", "./models/random_forest_heart_disease_v1.bin"]


EXPOSE 9696


ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
