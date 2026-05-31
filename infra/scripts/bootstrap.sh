#!/usr/bin/env bash
# ============================================================================
# 6G Digital Immunity - Infrastructure Bootstrap Script
# ============================================================================
# Generates mTLS certificates and prepares Docker secrets for the 3-tier
# isolated network topology:
#   telemetry-net : 172.28.0.0/16
#   inference-net : 172.29.0.0/16
#   actuation-net : 172.30.0.0/16
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CERT_DIR="${PROJECT_ROOT}/infra/certs"
CA_NAME="6G-DI-RootCA"
VALIDITY_DAYS=3650
KEY_SIZE=4096

# Component definitions: name -> (CN, IP list)
declare -A COMPONENT_CN=(
    ["dsf-server"]="dsf-server.inference-net.6g-di.local"
    ["llm-agent"]="llm-agent.inference-net.6g-di.local"
    ["osmocom-actuator"]="osmocom-actuator.actuation-net.6g-di.local"
)

declare -A COMPONENT_IPS=(
    ["dsf-server"]="172.29.0.20"
    ["llm-agent"]="172.29.0.30"
    ["osmocom-actuator"]="172.30.0.100"
)

# Kafka / Zookeeper component definitions
KAFKA_CN="kafka.inference-net.6g-di.local"
KAFKA_IP="172.29.0.10"
ZOOKEEPER_CN="zookeeper.inference-net.6g-di.local"
ZOOKEEPER_IP="172.29.0.11"

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ---------------------------------------------------------------------------
# Logging functions
# ---------------------------------------------------------------------------

_ts() {
    date '+%Y-%m-%d %H:%M:%S %Z'
}

log_info() {
    local msg="$*"
    echo -e "${BLUE}[$(_ts)] [INFO]  ${NC}${msg}"
}

log_error() {
    local msg="$*"
    echo -e "${RED}[$(_ts)] [ERROR] ${NC}${msg}" >&2
}

log_success() {
    local msg="$*"
    echo -e "${GREEN}[$(_ts)] [OK]    ${NC}${msg}"
}

log_warn() {
    local msg="$*"
    echo -e "${YELLOW}[$(_ts)] [WARN]  ${NC}${msg}"
}

log_step() {
    local step_num="$1"
    local total="$2"
    local msg="$3"
    echo -e "\n${CYAN}${BOLD}━━━ Step ${step_num}/${total}: ${msg} ${NC}"
}

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    if ! command -v openssl &>/dev/null; then
        missing+=("openssl")
    fi

    if ! command -v docker &>/dev/null; then
        missing+=("docker")
    else
        if ! docker info &>/dev/null 2>&1; then
            log_error "Docker daemon is not running or not accessible."
            missing+=("docker-daemon")
        fi
    fi

    if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
        missing+=("docker-compose")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing prerequisites: ${missing[*]}"
        log_error "Please install them before running this script."
        exit 1
    fi

    log_success "All prerequisites satisfied (openssl, docker, docker-compose)."
}

# ---------------------------------------------------------------------------
# Certificate generation functions
# ---------------------------------------------------------------------------

generate_ca() {
    log_info "Generating self-signed Root CA ..."

    local ca_dir="${CERT_DIR}/ca"
    mkdir -p "${ca_dir}"

    local ca_key="${ca_dir}/ca.key"
    local ca_cert="${ca_dir}/ca.crt"

    openssl genrsa \
        -out "${ca_key}" \
        "${KEY_SIZE}" \
        2>/dev/null

    openssl req -x509 -new -nodes \
        -key "${ca_key}" \
        -sha256 \
        -days "${VALIDITY_DAYS}" \
        -out "${ca_cert}" \
        -subj "/C=US/ST=California/O=6G Digital Immunity/OU=PKI Infrastructure/CN=${CA_NAME}" \
        -addext "basicConstraints=critical,CA:TRUE,pathlen:1" \
        -addext "keyUsage=critical,keyCertSign,cRLSign" \
        -addext "subjectKeyIdentifier=hash" \
        -addext "authorityKeyIdentifier=keyid:always,issuer" \
        2>/dev/null

    chmod 600 "${ca_key}"
    chmod 644 "${ca_cert}"

    log_success "Root CA created at ${ca_dir}/"
    log_info "  CA Cert : ${ca_cert}"
    log_info "  CA Key  : ${ca_key}"
}

generate_server_cert() {
    local cn="$1"
    shift
    local -a san_list=("$@")

    local cert_dir="${CERT_DIR}/$(echo "${cn}" | sed 's/\..*//' | tr '[:upper:]' '[:lower:]')"

    # For components with known subdirectory names
    case "${cn}" in
        *dsf-server*)    cert_dir="${CERT_DIR}/dsf-server" ;;
        *llm-agent*)      cert_dir="${CERT_DIR}/llm-agent" ;;
        *osmocom-actuator*) cert_dir="${CERT_DIR}/osmocom-actuator" ;;
        *kafka*)          cert_dir="${CERT_DIR}/kafka" ;;
        *zookeeper*)      cert_dir="${CERT_DIR}/zookeeper" ;;
    esac

    mkdir -p "${cert_dir}"

    local key="${cert_dir}/server.key"
    local csr="${cert_dir}/server.csr"
    local cert="${cert_dir}/server.crt"
    local ca_key="${CERT_DIR}/ca/ca.key"
    local ca_cert="${CERT_DIR}/ca/ca.crt"

    log_info "Generating server cert for CN=${cn}"

    # Build SAN extension
    local san_entries=()
    for san in "${san_list[@]}"; do
        if [[ "${san}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            san_entries+=("IP:${san}")
        else
            san_entries+=("DNS:${san}")
        fi
    done

    local san_ext="subjectAltName=$(IFS=','; echo "${san_entries[*]}")"

    # Generate private key
    openssl genrsa \
        -out "${key}" \
        "${KEY_SIZE}" \
        2>/dev/null

    # Generate CSR
    openssl req -new \
        -key "${key}" \
        -out "${csr}" \
        -subj "/C=US/ST=California/O=6G Digital Immunity/CN=${cn}" \
        2>/dev/null

    # Sign with CA
    openssl x509 -req \
        -in "${csr}" \
        -CA "${ca_cert}" \
        -CAkey "${ca_key}" \
        -CAcreateserial \
        -out "${cert}" \
        -days "${VALIDITY_DAYS}" \
        -sha256 \
        -extfile <(printf "basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n${san_ext}\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer") \
        2>/dev/null

    # Clean up CSR
    rm -f "${csr}"

    log_success "Server cert generated: ${cert_dir}/"
    log_info "  CN         : ${cn}"
    log_info "  SAN entries : ${san_entries[*]}"
}

generate_client_cert() {
    local cn="$1"

    local cert_dir="${CERT_DIR}/$(echo "${cn}" | sed 's/\..*//' | tr '[:upper:]' '[:lower:]')"

    case "${cn}" in
        *dsf-server*)    cert_dir="${CERT_DIR}/dsf-server" ;;
        *llm-agent*)      cert_dir="${CERT_DIR}/llm-agent" ;;
        *osmocom-actuator*) cert_dir="${CERT_DIR}/osmocom-actuator" ;;
        *kafka*)          cert_dir="${CERT_DIR}/kafka" ;;
        *zookeeper*)      cert_dir="${CERT_DIR}/zookeeper" ;;
    esac

    mkdir -p "${cert_dir}"

    local key="${cert_dir}/client.key"
    local csr="${cert_dir}/client.csr"
    local cert="${cert_dir}/client.crt"
    local ca_key="${CERT_DIR}/ca/ca.key"
    local ca_cert="${CERT_DIR}/ca/ca.crt"

    log_info "Generating client cert for CN=${cn}"

    # Generate private key
    openssl genrsa \
        -out "${key}" \
        "${KEY_SIZE}" \
        2>/dev/null

    # Generate CSR
    openssl req -new \
        -key "${key}" \
        -out "${csr}" \
        -subj "/C=US/ST=California/O=6G Digital Immunity/CN=${cn}-client" \
        2>/dev/null

    # Sign with CA — clientAuth EKU
    openssl x509 -req \
        -in "${csr}" \
        -CA "${ca_cert}" \
        -CAkey "${ca_key}" \
        -CAcreateserial \
        -out "${cert}" \
        -days "${VALIDITY_DAYS}" \
        -sha256 \
        -extfile <(printf "basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=clientAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer") \
        2>/dev/null

    rm -f "${csr}"

    log_success "Client cert generated: ${cert_dir}/"
}

generate_certs_for_component() {
    local component_name="$1"
    local cn="$2"
    shift 2
    local -a ip_list=("$@")

    local san_list=("DNS:${cn}" "DNS:localhost")
    for ip in "${ip_list[@]}"; do
        san_list+=("IP:${ip}")
    done

    generate_server_cert "${cn}" "${san_list[@]}"
    generate_client_cert "${cn}"
}

verify_cert_chain() {
    local cert="$1"
    local ca_cert="$2"

    log_info "Verifying certificate chain: ${cert} against ${ca_cert}"

    if openssl verify -CAfile "${ca_cert}" "${cert}" &>/dev/null; then
        log_success "Certificate chain verified: ${cert}"
        return 0
    else
        log_error "Certificate chain verification FAILED: ${cert}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Kafka / Zookeeper certificate generation
# ---------------------------------------------------------------------------

generate_kafka_certs() {
    log_info "Generating Kafka broker certificates..."

    local kafka_san=(
        "DNS:${KAFKA_CN}"
        "DNS:kafka"
        "DNS:localhost"
        "IP:${KAFKA_IP}"
        "IP:127.0.0.1"
    )
    generate_server_cert "${KAFKA_CN}" "${kafka_san[@]}"
    generate_client_cert "${KAFKA_CN}"

    log_info "Generating Zookeeper certificates..."

    local zk_san=(
        "DNS:${ZOOKEEPER_CN}"
        "DNS:zookeeper"
        "DNS:localhost"
        "IP:${ZOOKEEPER_IP}"
        "IP:127.0.0.1"
    )
    generate_server_cert "${ZOOKEEPER_CN}" "${zk_san[@]}"
    generate_client_cert "${ZOOKEEPER_CN}"
}

# ---------------------------------------------------------------------------
# Permissions and Docker secrets
# ---------------------------------------------------------------------------

set_permissions() {
    log_info "Setting file permissions..."

    # Private keys — owner read/write only
    while IFS= read -r -d '' keyfile; do
        chmod 600 "${keyfile}"
        log_info "  chmod 600 ${keyfile}"
    done < <(find "${CERT_DIR}" -type f -name '*.key' -print0)

    # Certificates — world readable
    while IFS= read -r -d '' certfile; do
        chmod 644 "${certfile}"
        log_info "  chmod 644 ${certfile}"
    done < <(find "${CERT_DIR}" -type f \( -name '*.crt' -o -name '*.pem' \) -print0)

    log_success "Permissions applied."
}

create_docker_secrets() {
    log_info "Creating Docker secrets for mTLS..."

    # Remove pre-existing secrets (idempotent)
    docker secret rm \
        ca-cert \
        ca-key \
        dsf-server-cert \
        dsf-server-key \
        dsf-client-cert \
        dsf-client-key \
        llm-agent-cert \
        llm-agent-key \
        llm-client-cert \
        llm-client-key \
        osmocom-actuator-cert \
        osmocom-actuator-key \
        osmocom-client-cert \
        osmocom-client-key \
        kafka-server-cert \
        kafka-server-key \
        kafka-client-cert \
        kafka-client-key \
        zookeeper-server-cert \
        zookeeper-server-key \
        2>/dev/null || true

    # CA
    docker secret create ca-cert "${CERT_DIR}/ca/ca.crt" 2>/dev/null || true
    docker secret create ca-key  "${CERT_DIR}/ca/ca.key"  2>/dev/null || true

    # DSF Server
    docker secret create dsf-server-cert "${CERT_DIR}/dsf-server/server.crt" 2>/dev/null || true
    docker secret create dsf-server-key  "${CERT_DIR}/dsf-server/server.key"  2>/dev/null || true
    docker secret create dsf-client-cert "${CERT_DIR}/dsf-server/client.crt" 2>/dev/null || true
    docker secret create dsf-client-key  "${CERT_DIR}/dsf-server/client.key"  2>/dev/null || true

    # LLM Agent
    docker secret create llm-agent-cert  "${CERT_DIR}/llm-agent/server.crt"  2>/dev/null || true
    docker secret create llm-agent-key   "${CERT_DIR}/llm-agent/server.key"   2>/dev/null || true
    docker secret create llm-client-cert "${CERT_DIR}/llm-agent/client.crt"  2>/dev/null || true
    docker secret create llm-client-key  "${CERT_DIR}/llm-agent/client.key"   2>/dev/null || true

    # Osmocom Actuator
    docker secret create osmocom-actuator-cert "${CERT_DIR}/osmocom-actuator/server.crt" 2>/dev/null || true
    docker secret create osmocom-actuator-key  "${CERT_DIR}/osmocom-actuator/server.key"  2>/dev/null || true
    docker secret create osmocom-client-cert   "${CERT_DIR}/osmocom-actuator/client.crt" 2>/dev/null || true
    docker secret create osmocom-client-key    "${CERT_DIR}/osmocom-actuator/client.key"   2>/dev/null || true

    # Kafka
    docker secret create kafka-server-cert "${CERT_DIR}/kafka/server.crt" 2>/dev/null || true
    docker secret create kafka-server-key  "${CERT_DIR}/kafka/server.key"  2>/dev/null || true
    docker secret create kafka-client-cert "${CERT_DIR}/kafka/client.crt" 2>/dev/null || true
    docker secret create kafka-client-key  "${CERT_DIR}/kafka/client.key"  2>/dev/null || true

    # Zookeeper
    docker secret create zookeeper-server-cert "${CERT_DIR}/zookeeper/server.crt" 2>/dev/null || true
    docker secret create zookeeper-server-key  "${CERT_DIR}/zookeeper/server.key"  2>/dev/null || true

    log_success "Docker secrets created."
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    echo -e "\n${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}║          6G Digital Immunity — PKI Bootstrap Complete          ║${NC}"
    echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${BOLD}Root CA:${NC}"
    echo -e "  Certificate : ${CYAN}${CERT_DIR}/ca/ca.crt${NC}"
    echo -e "  Private Key : ${CYAN}${CERT_DIR}/ca/ca.key${NC}"
    echo -e "  Validity    : ${VALIDITY_DAYS} days"
    echo -e "  Key Size    : ${KEY_SIZE} bits"
    echo ""

    echo -e "${BOLD}Component Certificates:${NC}"

    for component in dsf-server llm-agent osmocom-actuator kafka zookeeper; do
        local dir="${CERT_DIR}/${component}"
        local cn=""
        local ips=""
        case "${component}" in
            kafka)     cn="${KAFKA_CN}";     ips="${KAFKA_IP}" ;;
            zookeeper) cn="${ZOOKEEPER_CN}"; ips="${ZOOKEEPER_IP}" ;;
            *)         cn="${COMPONENT_CN[${component}]}"; ips="${COMPONENT_IPS[${component}]}" ;;
        esac

        echo -e ""
        echo -e "  ${BOLD}${component}${NC}  (CN: ${cn})"
        echo -e "    IPs : ${ips}"
        if [[ -f "${dir}/server.crt" ]]; then
            local expiry
            expiry=$(openssl x509 -in "${dir}/server.crt" -noout -enddate 2>/dev/null | cut -d= -f2)
            echo -e "    Server cert : ${GREEN}✓${NC} ${dir}/server.crt  expires ${expiry}"
        else
            echo -e "    Server cert : ${RED}✗${NC} not found"
        fi
        if [[ -f "${dir}/client.crt" ]]; then
            echo -e "    Client cert : ${GREEN}✓${NC} ${dir}/client.crt"
        else
            echo -e "    Client cert : ${RED}✗${NC} not found"
        fi
    done

    echo ""
    echo -e "${BOLD}Docker Secrets:${NC}"
    (docker secret ls --format "table {{.Name}}\t{{.CreatedAt}}" 2>/dev/null \
        | grep -E 'ca-|dsf-|llm-|osmocom-|kafka-|zookeeper-' \
        | while read -r line; do
            echo -e "  ${GREEN}✓${NC} ${line}"
        done) || true

    echo ""
    echo -e "${YELLOW}${BOLD}Next Steps:${NC}"
    echo -e "  1. Review certificates:  ${CYAN}openssl x509 -in <cert> -text -noout${NC}"
    echo -e "  2. Start services:       ${CYAN}docker compose -f docker-compose.yml up -d${NC}"
    echo -e "  3. Verify mTLS:          ${CYAN}openssl s_client -connect <host>:<port> -cert <client.crt> -key <client.key> -CAfile ca.crt${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local total_steps=8

    echo -e "\n${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║     6G Digital Immunity — Infrastructure Bootstrap (Phase 5)      ║${NC}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"

    # Step 1 — Prerequisites
    log_step 1 "${total_steps}" "Checking prerequisites"
    check_prerequisites

    # Step 2 — Create output directories
    log_step 2 "${total_steps}" "Creating output directories"
    for dir in ca dsf-server llm-agent osmocom-actuator kafka zookeeper; do
        mkdir -p "${CERT_DIR}/${dir}"
    done
    log_success "Directory tree created under ${CERT_DIR}/"

    # Step 3 — Generate CA
    log_step 3 "${total_steps}" "Generating Root CA"
    generate_ca

    # Step 4 — Generate component certificates
    log_step 4 "${total_steps}" "Generating component certificates"
    for component in dsf-server llm-agent osmocom-actuator; do
        local cn="${COMPONENT_CN[${component}]}"
        local ip="${COMPONENT_IPS[${component}]}"
        generate_certs_for_component "${component}" "${cn}" "${ip}"
    done

    # Step 5 — Generate Kafka/Zookeeper certs
    log_step 5 "${total_steps}" "Generating Kafka/Zookeeper certificates"
    generate_kafka_certs

    # Step 6 — Set permissions
    log_step 6 "${total_steps}" "Setting file permissions"
    set_permissions

    # Step 7 — Create Docker secrets
    log_step 7 "${total_steps}" "Creating Docker secrets"
    create_docker_secrets

    # Step 8 — Print summary
    log_step 8 "${total_steps}" "Verification and summary"
    local ca_cert="${CERT_DIR}/ca/ca.crt"
    for component in dsf-server llm-agent osmocom-actuator kafka zookeeper; do
        local server_cert="${CERT_DIR}/${component}/server.crt"
        if [[ -f "${server_cert}" ]]; then
            verify_cert_chain "${server_cert}" "${ca_cert}"
        fi
    done
    print_summary

    log_success "Bootstrap completed successfully."
}

main "$@"
