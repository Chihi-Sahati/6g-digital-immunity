# ============================================================
# Dockerfile - DSF Safety Filter (C++ Core + Python Wrapper + gRPC)
# ============================================================
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    libgrpc++-dev \
    protobuf-compiler \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Build CBF C++ library ---
FROM base AS cpp-build
WORKDIR /build
COPY safety-filter/CMakeLists.txt .
COPY safety-filter/core/ ./core/
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FOUND=OFF \
    && make -j$(nproc) \
    && mkdir -p /app/lib \
    && cp libcbf_core.a /app/lib/

# --- Stage 3: Final ---
FROM base AS final
COPY --from=cpp-build /app/lib/libcbf_core.a /app/lib/

# Copy application code
COPY safety-filter/ /app/safety-filter/
COPY schemas/ /app/schemas/
COPY osmo-bsc/config/ /app/osmo-bsc/config/
COPY osmo-bts/config/ /app/osmo-bts/config/

COPY api/ /app/api/
# Generate protobuf Python stubs
RUN python -m grpc_tools.protoc \
    -Ischemas \
    -Iapi/proto \
    --python_out=. \
    --grpc_python_out=. \
    schemas/network_state.proto \
    api/proto/llm_intent.proto

# Environment
ENV SAFETY_FILTER_LIB=/app/lib/libcbf_core.a
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 50052 50053

CMD ["python", "-m", "safety-filter.api.grpc_interceptor"]
