FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir confluent-kafka numpy

COPY ran_simulator.py /app/

CMD ["python", "ran_simulator.py"]
