FROM python:3.10-bullseye

WORKDIR /home/mlflow

RUN apt-get update && apt-get install -y --no-install-recommends \
            vim nano curl


COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN pip install virtualenv

#add envrionment variables from .env file
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_DEFAULT_REGION=""

RUN export MLFLOW_TRACKING_URI=http://localhost:5000

#Local
CMD ["mlflow" ,"ui" , "-h" , "0.0.0.0", "-p" , "5000"]

# S3
# CMD ["mlflow" ,"ui" , "-h" , "0.0.0.0" ,"--default-artifact-root", "s3://..../"]



