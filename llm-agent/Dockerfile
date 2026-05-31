# ============================================================
# Dockerfile - LLM Agent (TelcoLLM Inference + RCA + gRPC Client)
# ============================================================
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY llm-agent/ /app/llm-agent/
COPY safety-filter/core/cbf_engine.py /app/safety-filter/core/
COPY safety-filter/core/zeno_guard.py /app/safety-filter/core/
COPY schemas/ /app/schemas/
COPY osmo-bsc/config/ /app/osmo-bsc/config/
COPY osmo-bts/config/ /app/osmo-bts/config/

COPY api/ /app/api/
# Generate protobuf stubs
RUN python -m grpc_tools.protoc \
    -Ischemas \
    -Iapi/proto \
    --python_out=. \
    --grpc_python_out=. \
    schemas/network_state.proto \
    api/proto/llm_intent.proto

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DSF_HOST=172.29.0.20
ENV DSF_PORT=50052

CMD ["python", "-m", "llm-agent.api.grpc_client"]
