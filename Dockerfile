FROM python:3.10-bullseye

WORKDIR /home/mlflow

RUN apt-get update && apt-get install -y --no-install-recommends \
            vim nano curl


COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN curl https://pyenv.run | bash

RUN echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc

RUN pip install virtualenv

RUN export MLFLOW_TRACKING_URI=http://localhost:5000

CMD ["mlflow" ,"ui" , "-h" , "0.0.0.0", "-p" , "5000"]
