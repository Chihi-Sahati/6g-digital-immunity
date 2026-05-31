# ============================================================
# Dockerfile - Syslog Forwarder (Osmocom Telemetry -> Kafka)
# ============================================================
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    syslog-ng-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY telemetry/ /app/telemetry/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "telemetry.collectors.syslog_forwarder", \
     "--mode", "tail", \
     "--kafka-bootstrap", "172.29.0.10:9092", \
     "--topic", "telecomm.bts.telemetry", \
     "--log-source", "/var/log/osmocom/"]
