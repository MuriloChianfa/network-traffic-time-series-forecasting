FROM python:3.11.10-slim

EXPOSE 8501
WORKDIR /app

RUN apt update && apt -o Acquire::ForceIPv4=true -y \
    --allow-unauthenticated install -f \
    software-properties-common \
    build-essential \
    curl \
    git \
    && apt clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./forecasting/ /app
RUN mkdir /datasets
COPY ./datasets/ /datasets

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
