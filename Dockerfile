FROM python:3.10-bullseye

WORKDIR /home/mlflow

RUN apt-get update && apt-get install -y --no-install-recommends \
            vim nano curl


COPY ./requirements.txt .
RUN pip install -r requirements.txt


CMD ["mlflow" ,"ui" , "-h" , "0.0.0.0", "-p" , "5000"]
