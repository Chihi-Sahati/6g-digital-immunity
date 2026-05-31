# Reproducibility Guide — 6G Digital Immunity Framework
# ========================================================
# This document provides step-by-step instructions for reproducing all
# experimental results reported in the manuscript:
#   "Deterministic Digital Immunity for 6G Networks: A CBF-Guided
#    LLM Agent Architecture"
# ========================================================================

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Quick Start (Docker)](#2-quick-start-docker)
3. [Running Components Independently](#3-running-components-independently)
4. [Reproducing Paper Results](#4-reproducing-paper-results)
5. [Expected Outputs](#5-expected-outputs)
6. [Known Limitations](#6-known-limitations)
7. [Troubleshooting Guide](#7-troubleshooting-guide)

---

## 1. System Requirements

### Hardware

| Resource        | Minimum       | Recommended   |
|-----------------|---------------|---------------|
| CPU             | 4 cores       | 8 cores       |
| RAM             | 8 GB          | 16 GB         |
| Disk Space      | 20 GB         | 40 GB         |
| Network         | LAN           | Stable internet (for LLM API) |

### Software

| Component      | Version       | Notes                                 |
|----------------|---------------|---------------------------------------|
| Docker         | >= 24.0       | Docker Engine or Desktop              |
| Docker Compose | >= 2.20       | `docker compose` (v2 plugin)          |
| Python         | 3.11 or 3.12  | Used inside containers and for eval   |
| Git            | >= 2.30       | For cloning the repository            |
| OpenSSL        | >= 3.0        | For mTLS certificate generation       |
| pip            | >= 23.0       | Python package manager                |

### Operating Systems

Verified on:
- **Ubuntu 22.04 LTS** (primary development environment)
- **macOS 13+** (with Docker Desktop)
- **Windows 11** (with WSL2 + Docker Desktop)

> **Note**: Docker is required. The system cannot run natively without
> Docker due to the multi-service architecture (Kafka, Zookeeper,
> Prometheus, etc.).

---

## 2. Quick Start (Docker)

### 2.1 Clone the Repository

```bash
git clone https://github.com/<organization>/6g-digital-immunity.git
cd 6g-digital-immunity
```

### 2.2 Generate mTLS Certificates

The framework uses mutual TLS (mTLS) for all inter-service communication.
Run the bootstrap script to generate the full PKI infrastructure:

```bash
bash ./infra/scripts/bootstrap.sh
```

This generates:
- Root CA: `infra/certs/ca/ca.crt`, `infra/certs/ca/ca.key`
- Component server + client certificates for each service
- Docker secrets for container deployment

**Expected output**: A summary table listing all generated certificates
with their expiry dates.

### 2.3 Build and Launch All Services

```bash
docker compose up -d --build
```

This builds and starts all 8 services:

| Service            | Container Name         | Network(s)         | Port(s)    |
|--------------------|------------------------|--------------------|------------|
| RAN Simulator      | 6gdi-osmo-bsc         | telemetry, inference | —        |
| OsmoBTS            | 6gdi-osmo-bts         | actuation          | 4243:4242  |
| Syslog Forwarder   | 6gdi-syslog-forwarder | telemetry, inference | —      |
| Kafka              | 6gdi-kafka            | inference          | 9092, 9093 |
| Zookeeper          | 6gdi-zookeeper        | inference          | 2181       |
| DSF Server         | 6gdi-dsf-server       | inference          | 50052, 50053 |
| LLM Agent          | 6gdi-llm-agent        | inference          | —          |
| Network Actuator   | 6gdi-network-actuator | inference, actuation | 50054   |
| Prometheus         | 6gdi-prometheus       | telemetry          | 9090       |

### 2.4 Verify Services Are Running

```bash
# Check all container statuses
docker compose ps

# Verify DSF gRPC endpoint
docker exec 6gdi-dsf-server python -c "print('DSF OK')"

# Verify LLM Agent
docker exec 6gdi-llm-agent python -c "print('LLM Agent OK')"

# Check Kafka is responsive
docker exec 6gdi-kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Prometheus metrics
curl -s http://localhost:9090/-/healthy
```

### 2.5 Run the Integration Test

Submit a test intent through the DSF safety filter:

```bash
docker cp test_intent.py 6gdi-llm-agent:/app/test_intent.py
docker exec 6gdi-llm-agent python /app/test_intent.py
```

**Expected output**:
```
Connecting to DSF Server on dsf-server:50052...
Submitting intent to DSF Server for validation...

=== VERDICT RECEIVED ===
Decision: ALLOW
Verdict ID: verdict-...
Min Margin: ...
```

### 2.6 Teardown

```bash
# Stop all services and remove volumes
docker compose down -v
```

---

## 3. Running Components Independently

Each component can be developed and tested in isolation.

### 3.1 RAN Simulator (Standalone)

The RAN simulator publishes synthetic telemetry to Kafka using
Ornstein-Uhlenbeck stochastic processes.

```bash
# Build only the simulator
docker compose build osmo-bsc

# Start with Kafka dependency
docker compose up -d zookeeper kafka osmo-bsc

# View live telemetry
docker compose logs -f osmo-bsc
```

**Simulated metrics**:
- `radio.prb_utilisation_pct` — PRB utilization (%) [0, 100]
- `radio.active_ue_count` — Active user equipment count
- `radio.rsrp` — Reference Signal Received Power (dBm)
- `radio.sinr` — Signal-to-Interference-plus-Noise Ratio (dB)
- `radio.tx_power` — Transmit power (dBm)

### 3.2 DSF Safety Filter (Standalone)

The Deterministic Safety Filter (DSF) evaluates CBF constraints and
issues ALLOW / AMEND / DENY verdicts.

```bash
# Build the DSF server
docker compose build dsf-server

# Start with dependencies
docker compose up -d zookeeper kafka dsf-server

# Test via the intent test script
docker cp test_intent.py 6gdi-dsf-server:/app/test_intent.py
docker exec 6gdi-dsf-server python /app/test_intent.py
```

**CBF Barriers Enforced**:

| ID  | Barrier Function              | Safe Set                        |
|-----|-------------------------------|---------------------------------|
| h₁  | 43 − tx_power                 | tx_power ≤ 43 dBm              |
| h₂  | tx_power − 0                  | tx_power ≥ 0 dBm               |
| h₃  | 100 − prb_utilization         | prb_utilization ≤ 100%         |
| h₄  | ho_hysteresis − 0             | ho_hysteresis ≥ 0 dB           |

### 3.3 LLM Agent (Standalone)

The TelcoLLM agent generates network optimization suggestions using
free LLM APIs (via the `g4f` library).

```bash
# Build the LLM agent
docker compose build llm-agent

# Start with dependencies
docker compose up -d zookeeper kafka dsf-server llm-agent

# View agent logs
docker compose logs -f llm-agent
```

### 3.4 Network Actuator (Standalone)

```bash
docker compose build network-actuator
docker compose up -d osmo-bts network-actuator
docker compose logs -f network-actuator
```

### 3.5 Prometheus Monitoring

```bash
docker compose up -d prometheus

# Access the Prometheus UI
# Open http://localhost:9090 in your browser
```

---

## 4. Reproducing Paper Results

### 4.1 Reproduce CBF Safety Margin Plots (Figure X)

The CBF safety margin plots demonstrate that all barrier functions
maintain h(x) ≥ 0 across all intents, validating the 100% safety claim.

```bash
# Install evaluation dependencies locally
pip install matplotlib seaborn numpy scipy

# Generate the plots
cd evaluation
python plot_cbf_safety_margins.py --runs 10 --intents 500 --seed 42
```

**Output files**:
- `evaluation/cbf_safety_margins.pdf` — Vector format for LaTeX
- `evaluation/cbf_safety_margins.png` — Raster at 300 DPI

**What the script produces**:
1. **(a) Time Series** — h(x) values for all 4 barriers across 500 intents
2. **(b) Distribution** — Violin + box plots of safety margins per barrier
3. **(c) Minimum Margin** — Per-run minimum with 95% CI across 10 runs
4. **(d) Margin vs. Acceptance Rate** — Scatter plot with regression

### 4.2 Reproduce RAN Telemetry Plots (Figure Y)

The RAN telemetry plots visualize PRB utilization and active users
over time, showing the traffic dynamics generated by the O-U process.

```bash
# Step 1: Start the RAN simulator and collect metrics
docker compose up -d zookeeper kafka osmo-bsc

# Wait for data collection (~60 seconds recommended)
sleep 60

# Step 2: Collect metrics from Kafka
# (Use collect_metrics.py or the Kafka consumer API)
cd evaluation
python collect_metrics.py

# Step 3: Generate plots
python plot_results.py
```

**Output files**:
- `evaluation/ran_telemetry.pdf`
- `evaluation/ran_telemetry.png`

### 4.3 Reproduce Intent Validation Results

To reproduce the intent acceptance / amendment / denial statistics:

```bash
# Ensure full stack is running
docker compose up -d --build

# Submit multiple test intents with varying parameters
# (Modify test_intent.py to iterate over parameter ranges)
docker cp test_intent.py 6gdi-llm-agent:/app/test_intent.py
docker exec 6gdi-llm-agent python /app/test_intent.py

# Collect audit logs from Kafka topic "safety-audit-log"
```

**Paper-reported metrics**:
- Safety guarantee: 100% (h(x) ≥ 0 for all barriers)
- Intent acceptance rate: ~95–98% (safe intents pass)
- Intent amendment rate: ~2–5% (near-boundary intents get clamped)
- Intent denial rate: ~0% (only unsafe intents are denied)
- DSF evaluation latency: < 5 ms per intent

### 4.4 Custom Simulation Parameters

```bash
# Generate plots with different parameters
python evaluation/plot_cbf_safety_margins.py \
    --runs 20 \
    --intents 1000 \
    --seed 12345 \
    --output-dir ./custom_output
```

---

## 5. Expected Outputs

### 5.1 CBF Safety Margin Plots

| Subplot     | Description                                             |
|-------------|---------------------------------------------------------|
| (a)         | Time series of h₁–h₄ across 500 intents                |
| (b)         | Violin + box distribution of safety margins             |
| (c)         | Per-run minimum margin with 95% confidence interval     |
| (d)         | Scatter: minimum margin vs. intent acceptance rate       |

Key observations to verify:
- All h(x) values remain ≥ 0 (no safety violations)
- h₁ (tx power ceiling) has margins in the range ~3–43 dBm
- h₂ (tx power floor) has margins in the range ~2–40 dBm
- h₃ (PRB utilization ceiling) has margins in the range ~15–90%
- h₄ (HO hysteresis floor) has margins in the range ~0.5–4 dB

### 5.2 Console Output Summary

When running `plot_cbf_safety_margins.py`, expect:

```
=================================================================
  CBF Safety Margin Summary Statistics
=================================================================
  h₁  (Tx Power Ceiling: 43 dBm)
      min=  3.0000  max= 41.0000  mean= 13.0000  std=  4.0000
      Safety violations: 0 / 5000 (0.0000%)
  h₂  (Tx Power Floor: 0 dBm)
      min=  2.0000  max= 40.0000  mean= 30.0000  std=  4.0000
      Safety violations: 0 / 5000 (0.0000%)
  h₃  (PRB Utilization Ceiling: 100%)
      min= 15.0000  max= 90.0000  mean= 47.5000  std= 18.7500
      Safety violations: 0 / 5000 (0.0000%)
  h₄  (HO Hysteresis Floor: 0 dB)
      min=  0.5000  max=  4.0000  mean=  2.0000  std=  0.5000
      Safety violations: 0 / 5000 (0.0000%)

  Overall acceptance rate: 100.00% (± 0.00%)
=================================================================
```

### 5.3 File Structure After Full Run

```
6g-digital-immunity/
├── evaluation/
│   ├── cbf_safety_margins.pdf      # ← Generated CBF plots
│   ├── cbf_safety_margins.png      # ← Generated CBF plots
│   ├── ran_telemetry.pdf           # ← Generated telemetry plots
│   ├── ran_telemetry.png           # ← Generated telemetry plots
│   ├── metrics_log.json            # ← Collected metrics data
│   ├── plot_cbf_safety_margins.py
│   ├── plot_results.py
│   └── collect_metrics.py
├── infra/certs/                    # ← Generated mTLS certificates
│   ├── ca/
│   ├── dsf-server/
│   ├── llm-agent/
│   ├── osmocom-actuator/
│   ├── kafka/
│   └── zookeeper/
└── ...
```

---

## 6. Known Limitations

### 6.1 g4f LLM API Instability

The `g4f` (GPT4Free) library used by the LLM agent relies on
unofficial, reverse-engineered APIs from various LLM providers.

**Known issues**:
- APIs may be rate-limited, blocked, or change without notice
- Response quality and availability vary by provider
- Some providers require CAPTCHA solving or browser fingerprinting
- The `curl_cffi` dependency handles TLS fingerprint spoofing but
  may break when upstream providers update their defenses

**Mitigation strategies**:
1. **Substitute with a real LLM API**: Replace `g4f` with OpenAI,
   Anthropic, or a local model (e.g., Ollama, vLLM) for stable results.
2. **Use mock mode**: The DSF safety filter operates independently
   of the LLM. Test CBF guarantees without the LLM agent running.
3. **Pin versions**: Use `g4f==0.3.0` and `curl_cffi==0.6.0` as
   specified in `requirements.txt`.

### 6.2 Docker Resource Constraints

On systems with limited RAM (< 8 GB), the full Docker Compose stack
may experience OOM kills. Monitor with:

```bash
docker stats
```

If needed, reduce resource limits in `docker-compose.yml` or run
services selectively (see Section 3).

### 6.3 C++ CBF Library Compilation

The CBF math library (`safety-filter/core/cbf_math.cpp`) requires:
- CMake >= 3.15
- Eigen3 headers
- A C++17 compatible compiler (GCC >= 9, Clang >= 10)

On some systems, the Eigen3 package may not be available. Install with:
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen
```

### 6.4 Simulation Determinism

The RAN simulator uses `numpy.random` for stochastic processes. Results
are reproducible when the random seed is fixed, but will vary across
different NumPy versions due to implementation differences in the
random number generator.

### 6.5 Network Isolation Limitations

The 3-tier network isolation (iptables rules) described in the
architecture is enforced at the Docker network level. On macOS and
Windows, Docker Desktop runs inside a VM, so iptables rules inside
containers may behave differently than on native Linux.

---

## 7. Troubleshooting Guide

### Problem: `docker compose up` fails with certificate errors

**Cause**: mTLS certificates not generated or paths are incorrect.

**Solution**:
```bash
# Regenerate all certificates
rm -rf infra/certs/
bash ./infra/scripts/bootstrap.sh
docker compose up -d --build
```

### Problem: Kafka is unreachable (`connection refused`)

**Cause**: Zookeeper not yet healthy when Kafka starts.

**Solution**:
```bash
# Restart Kafka after Zookeeper is ready
docker compose restart kafka
# Or wait and retry
sleep 30
docker compose logs kafka
```

### Problem: DSF gRPC returns `UNAVAILABLE`

**Cause**: DSF server not yet listening on port 50052.

**Solution**:
```bash
# Check DSF logs for startup errors
docker compose logs dsf-server

# Verify port is listening
docker exec 6gdi-dsf-server python -c \
    "import socket; s = socket.socket(); s.connect(('localhost', 50052)); s.close()"
```

### Problem: LLM agent produces no suggestions

**Cause**: `g4f` API is blocked or rate-limited (see Section 6.1).

**Solution**:
```bash
# Check LLM agent logs
docker compose logs llm-agent

# Test g4f connectivity locally
pip install g4f curl_cffi
python -c "
import g4f
resp = g4f.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'Hello'}],
)
print(resp)
"
```

If this fails, substitute with a local model or mock the LLM responses.

### Problem: `plot_cbf_safety_margins.py` fails with `Font not found`

**Cause**: Times New Roman font not installed on the system.

**Solution**:
```bash
# Install msttcorefonts (Ubuntu/Debian)
sudo apt-get install msttcorefonts -y
fc-cache -fv

# Or install manually
mkdir -p ~/.local/share/fonts
# Download Times New Roman TTF and place in the fonts directory
```

Alternatively, the script will fall back to the system default serif
font (`DejaVu Serif`), which is acceptable for most purposes.

### Problem: `pytest` cannot find tests

**Cause**: No `tests/` directory exists yet (tests are being developed).

**Solution**: Tests will be added in a future update. For now, run
the integration test via Docker:
```bash
docker compose up -d --build
docker cp test_intent.py 6gdi-llm-agent:/app/test_intent.py
docker exec 6gdi-llm-agent python /app/test_intent.py
```

### Problem: `mypy` reports many errors

**Cause**: The codebase uses `from __future__ import annotations` and
some dynamic typing patterns. Type stubs for `grpc`, `kafka`, and `g4f`
may not be fully available.

**Solution**: Run mypy with `--ignore-missing-imports` as shown in the
CI configuration (`.github/workflows/test.yml`). Address specific type
errors incrementally.

### Problem: Docker build fails with `pip install` timeout

**Cause**: Network connectivity issues or PyPI rate limiting.

**Solution**:
```bash
# Use a mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or increase timeout
docker compose build --build-arg PIP_TIMEOUT=300
```

### Problem: Port already in use

**Cause**: Another service is using ports 9090, 9092, 50052, etc.

**Solution**: Modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "19090:9090"  # Prometheus on alternative port
  - "19092:9092"  # Kafka on alternative port
```

> **Note**: Internal container-to-container communication uses the
> Docker network IPs and is unaffected by host port mappings.

---

## Contact & Support

For issues not covered in this guide, please open a GitHub Issue on the
repository with the following information:
- Operating system and version
- Docker version (`docker --version`)
- Full error output
- Relevant container logs (`docker compose logs <service>`)
