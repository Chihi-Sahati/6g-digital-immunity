# ============================================================
# Dockerfile - Network Actuator (DSF -> Osmocom VTY Bridge)
# ============================================================
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    telnet \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY safety-filter/actuation/ /app/safety-filter/actuation/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OSMO_VTY_HOST=172.30.0.100
ENV OSMO_VTY_PORT=4242

CMD ["python", "-m", "safety-filter.actuation.network_actuator", \
     "--host", "0.0.0.0", "--port", "50054"]
