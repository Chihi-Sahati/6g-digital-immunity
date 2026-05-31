#!/usr/bin/env python3
"""
6G Digital Immunity — Syslog Forwarder Pipeline (Phase 5)
==========================================================
Taps Osmocom log streams, parses radio/bearer/event metrics from raw syslog
lines, transforms them into NetworkState-compatible telemetry points, and
publishes to Kafka for downstream inference consumption.

Designed to run inside the telemetry-net container, bridging telemetry
collection to the inference-net Kafka broker.

Usage:
    # Live tail mode (follow log files)
    python syslog_forwarder.py --config /etc/forwarder/config.json

    # Batch replay mode (read existing files)
    python syslog_forwarder.py --config /etc/forwarder/config.json --mode batch \
        --input /var/log/osmocom/bsc.log

    # Single log source override
    python syslog_forwarder.py \
        --log-source /var/log/osmocom/bsc.log \
        --element-id BSC-001 \
        --element-label OsmoBSC-Primary \
        --kafka-bootstrap kafka:9092 \
        --kafka-topic network-telemetry

Environment variables (override config):
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_TELEMETRY, KAFKA_TOPIC_EVENTS,
    KAFKA_CLIENT_ID, LOG_SOURCES, FORWARDER_MODE, BATCH_SIZE,
    FLUSH_INTERVAL_SEC, PARSE_RADIO_METRICS, PARSE_BEARER_METRICS, PARSE_EVENTS
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("syslog_forwarder")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SyslogEvent:
    """Parsed representation of a single syslog line."""

    timestamp: str
    source_host: str
    source_program: str
    severity: str
    message: str
    raw_line: str
    parsed_fields: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TelemetryPoint:
    """A single telemetry data point matching network_state.proto schema."""

    element_id: str
    element_label: str
    metric_name: str
    metric_value: float
    unit: str
    timestamp: str
    source: str
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "element_id": self.element_id,
            "element_label": self.element_label,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "source": self.source,
            **self.extra,
        }


@dataclasses.dataclass
class LogSource:
    """Configuration for a single log source to tail."""

    path: str
    element_id: str
    element_label: str


# ---------------------------------------------------------------------------
# Regex patterns for Osmocom log parsing
# ---------------------------------------------------------------------------

# Standard BSD / RFC 3164 syslog line:
#   May 15 14:32:01 hostname program[pid]: <level> message
_SYSLOG_RE = re.compile(
    r"(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<program>[a-zA-Z0-9_.-]+?)(?:\[(?P<pid>\d+)\])?\s*:\s*"
    r"(?:(?P<severity>\w+)\s+)?"
    r"(?P<message>.*)"
)

# RFC 5424 syslog:
#   <priority>1 version timestamp hostname app-name procid msgid structured-data msg
_SYSLOG5424_RE = re.compile(
    r"<(?P<pri>\d+)>(?P<version>\d+)\s+"
    r"(?P<timestamp>\S+)\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<app>\S+)\s+"
    r"(?P<procid>\S+)\s+"
    r"(?P<msgid>-)\s+"
    r"(?:\[(?P<sd>.*?)\]\s+)?"
    r"(?P<message>.*)"
)

# Osmocom internal log levels
_OSMO_LEVEL_RE = re.compile(
    r"<(?P<level>(?:LOGL|LOG[DEINW])|D(?:EBUG)?|I(?:NFO|NFO)?|W(?:ARNING|ARN)?|E(?:RROR|RR)?)>"
)

# Radio metric patterns
_RSRP_RE = re.compile(r"(?:RSRP|rsrp)\s*[:=]\s*-?\d+(?:\.\d+)?\s*(?:dBm)?")
_RSRQ_RE = re.compile(r"(?:RSRQ|rsrq)\s*[:=]\s*-?\d+(?:\.\d+)?\s*(?:dB)?")
_RSSI_RE = re.compile(r"(?:RSSI|rssi)\s*[:=]\s*-?\d+(?:\.\d+)?\s*(?:dBm)?")
_SINR_RE = re.compile(r"(?:SINR|sinr)\s*[:=]\s*-?\d+(?:\.\d+)?\s*(?:dB)?")
_POWER_LEVEL_RE = re.compile(
    r"(?:Tx|TX|tx|Power|POWER)\s*(?:Power|power|Level|level)?\s*[:=]\s*-?\d+(?:\.\d+)?\s*(?:dBm)?"
)
_ARFCN_RE = re.compile(r"(?:ARFCN|arfcn)\s*[:=]\s*(\d+)")
_Band_RE = re.compile(r"(?:Band|BAND)\s*[:=]\s*(\S+)")
_CELL_ID_RE = re.compile(r"(?:Cell ID|cell_id|CI)\s*[:=]\s*(\S+)")
_TAC_RE = re.compile(r"(?:TAC|tac)\s*[:=]\s*(\d+)")
_PCI_RE = re.compile(r"(?:PCI|pci)\s*[:=]\s*(\d+)")

# Bearer / GTP metrics
_GTP_TEID_RE = re.compile(r"(?:TEID|teid)\s*[:=]\s*(0x[\da-fA-F]+|\d+)")
_GTP_BEARER_RE = re.compile(r"(?:Bearer|bearer|EPS bearer)\s*(?:ID|id)?\s*[:=]\s*(\d+)")
_THROUGHPUT_DL_RE = re.compile(
    r"(?:DL|dl|downlink|Downlink)\s*(?:throughput|bitrate|rate)?\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:(k|m|M|G)bps|kbps|Mbps|Gbps)?"
)
_THROUGHPUT_UL_RE = re.compile(
    r"(?:UL|ul|uplink|Uplink)\s*(?:throughput|bitrate|rate)?\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:(k|m|M|G)bps|kbps|Mbps|Gbps)?"
)
_QCI_RE = re.compile(r"(?:QCI|qci)\s*[:=]\s*(\d+)")
_ARP_RE = re.compile(r"(?:ARP|arp)\s*[:=]\s*(\d+)")

# Event patterns
_HO_EVENT_RE = re.compile(
    r"(?:handover|HO)\s*(?:event|start|complete|failure|trigger|request|command|prepare|execute)",
    re.IGNORECASE,
)
_CELL_RESEL_RE = re.compile(
    r"(?:cell reselection|reselection)",
    re.IGNORECASE,
)
_UE_ATTACH_RE = re.compile(
    r"(?:UE|ue)\s*(?:attach| Attach|detach| Detach|registration|Registration|de-registration)",
    re.IGNORECASE,
)
_BEARER_SETUP_RE = re.compile(
    r"(?:bearer|Bearer)\s*(?:setup| Setup|establish| Establish|create|release|Release|modify|Modify)",
    re.IGNORECASE,
)
_RACH_RE = re.compile(
    r"(?:RACH|rach)\s*(?:attempt|request|indication|procedure)", re.IGNORECASE
)
_PAGING_RE = re.compile(r"(?:paging|Paging)", re.IGNORECASE)

# Numeric value extraction
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Osmocom Syslog Parser
# ---------------------------------------------------------------------------


class OsmocomSyslogParser:
    """Parses raw Osmocom syslog lines into structured events."""

    def __init__(self, element_id: str, element_label: str) -> None:
        self.element_id = element_id
        self.element_label = element_label

    def parse_line(self, line: str) -> Optional[SyslogEvent]:
        """Parse a single syslog line into a SyslogEvent, or None if unparseable."""
        line = line.rstrip("\n\r")

        # Attempt RFC 5424
        m = _SYSLOG5424_RE.match(line)
        if m:
            severity = self._pri_to_severity(int(m.group("pri")))
            return SyslogEvent(
                timestamp=m.group("timestamp"),
                source_host=m.group("host"),
                source_program=m.group("app"),
                severity=severity,
                message=m.group("message").strip(),
                raw_line=line,
                parsed_fields={},
            )

        # Attempt RFC 3164
        m = _SYSLOG_RE.match(line)
        if m:
            severity = m.group("severity") or "INFO"
            return SyslogEvent(
                timestamp=m.group("timestamp"),
                source_host=m.group("host"),
                source_program=m.group("program"),
                severity=severity,
                message=m.group("message").strip(),
                raw_line=line,
                parsed_fields={},
            )

        # Osmocom compact format (timestamp program: <level> message)
        m = _OSMO_LEVEL_RE.match(line)
        if m:
            return SyslogEvent(
                timestamp=datetime.now(timezone.utc).strftime("%b %d %H:%M:%S"),
                source_host=self.element_label,
                source_program="osmocom",
                severity=m.group("level"),
                message=line[m.end() :].strip(),
                raw_line=line,
                parsed_fields={},
            )

        return None

    # ------------------------------------------------------------------
    # Radio metric extraction
    # ------------------------------------------------------------------

    def parse_radio_metrics(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract RSRP, RSRQ, RSSI, SINR, power levels, ARFCN, etc."""
        fields: Dict[str, Any] = {}

        def _extract(pattern: str, key: str, cast: type = float) -> None:
            m = re.search(pattern, message)
            if m:
                num_match = _NUMERIC_RE.search(m.group(0))
                if num_match:
                    fields[key] = cast(num_match.group(0))

        _extract(r"(?:RSRP|rsrp)\s*[:=]\s*", "rsrp")
        _extract(r"(?:RSRQ|rsrq)\s*[:=]\s*", "rsrq")
        _extract(r"(?:RSSI|rssi)\s*[:=]\s*", "rssi")
        _extract(r"(?:SINR|sinr)\s*[:=]\s*", "sinr")
        _extract(r"(?:Tx|TX|tx)\s*(?:Power|power|Level|level)?\s*[:=]\s*", "tx_power")
        _extract(r"(?:Rx|RX|rx)\s*(?:Power|power|Level|level)?\s*[:=]\s*", "rx_power")

        m = _ARFCN_RE.search(message)
        if m:
            fields["arfcn"] = int(m.group(1))

        m = _Band_RE.search(message)
        if m:
            fields["band"] = m.group(1)

        m = _CELL_ID_RE.search(message)
        if m:
            fields["cell_id"] = m.group(1)

        m = _TAC_RE.search(message)
        if m:
            fields["tac"] = int(m.group(1))

        m = _PCI_RE.search(message)
        if m:
            fields["pci"] = int(m.group(1))

        return fields if fields else None

    # ------------------------------------------------------------------
    # Bearer metric extraction
    # ------------------------------------------------------------------

    def parse_bearer_metrics(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract GTP-U, bearer setup/release, QCI, throughput, etc."""
        fields: Dict[str, Any] = {}

        m = _GTP_TEID_RE.search(message)
        if m:
            fields["gtp_teid"] = m.group(1)

        m = _GTP_BEARER_RE.search(message)
        if m:
            fields["bearer_id"] = int(m.group(1))

        m = _THROUGHPUT_DL_RE.search(message)
        if m:
            val = float(m.group(1))
            multiplier = m.group(2)
            if multiplier and multiplier.lower() == "m":
                val *= 1e6
            elif multiplier and multiplier.lower() == "g":
                val *= 1e9
            elif multiplier and multiplier.lower() == "k":
                val *= 1e3
            fields["dl_throughput_bps"] = val

        m = _THROUGHPUT_UL_RE.search(message)
        if m:
            val = float(m.group(1))
            multiplier = m.group(2)
            if multiplier and multiplier.lower() == "m":
                val *= 1e6
            elif multiplier and multiplier.lower() == "g":
                val *= 1e9
            elif multiplier and multiplier.lower() == "k":
                val *= 1e3
            fields["ul_throughput_bps"] = val

        m = _QCI_RE.search(message)
        if m:
            fields["qci"] = int(m.group(1))

        m = _ARP_RE.search(message)
        if m:
            fields["arp"] = int(m.group(1))

        return fields if fields else None

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def parse_event(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse handover, cell reselection, UE attach/detach events."""
        events: List[str] = []

        if _HO_EVENT_RE.search(message):
            # Classify HO sub-type
            if re.search(r"complete|success", message, re.IGNORECASE):
                events.append("handover_complete")
            elif re.search(r"fail|error|reject", message, re.IGNORECASE):
                events.append("handover_failure")
            elif re.search(r"start|trigger|initiat", message, re.IGNORECASE):
                events.append("handover_start")
            elif re.search(r"prepare|request", message, re.IGNORECASE):
                events.append("handover_prepare")
            else:
                events.append("handover_other")
            # Extract target cell if available
            m = re.search(r"(?:target|Target)\s*(?:cell|Cell)\s*[:=]?\s*(\S+)", message)
            if m:
                return {"event_type": events[0], "target_cell": m.group(1)}
            return {"event_type": events[0]}

        if _CELL_RESEL_RE.search(message):
            m = re.search(
                r"(?:from|From)\s*(?:cell|Cell)\s*[:=]?\s*(\S+).*?(?:to|To)\s*(?:cell|Cell)\s*[:=]?\s*(\S+)",
                message,
            )
            if m:
                return {
                    "event_type": "cell_reselection",
                    "from_cell": m.group(1),
                    "to_cell": m.group(2),
                }
            return {"event_type": "cell_reselection"}

        if _UE_ATTACH_RE.search(message):
            if re.search(r"detach|de-regist", message, re.IGNORECASE):
                return {"event_type": "ue_detach"}
            return {"event_type": "ue_attach"}

        if _BEARER_SETUP_RE.search(message):
            if re.search(r"release|remove", message, re.IGNORECASE):
                return {"event_type": "bearer_release"}
            if re.search(r"modify", message, re.IGNORECASE):
                return {"event_type": "bearer_modify"}
            return {"event_type": "bearer_setup"}

        if _RACH_RE.search(message):
            return {"event_type": "rach_procedure"}

        if _PAGING_RE.search(message):
            return {"event_type": "paging"}

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pri_to_severity(pri: int) -> str:
        """Map RFC 5424 priority integer to severity name."""
        severity_map = {
            0: "EMERG",
            1: "ALERT",
            2: "CRIT",
            3: "ERROR",
            4: "WARNING",
            5: "NOTICE",
            6: "INFO",
            7: "DEBUG",
        }
        return severity_map.get(pri % 8, "INFO")


# ---------------------------------------------------------------------------
# Syslog tailer
# ---------------------------------------------------------------------------


class SyslogTailer:
    """Tails a log file or pipe, yielding lines as they arrive."""

    def __init__(self, log_source: str, element_id: str) -> None:
        self.log_source = log_source
        self.element_id = element_id
        self._process: Optional[subprocess.Popen] = None

    def tail(self) -> Generator[str, None, None]:
        """
        Follow a log source using ``tail -F``. Yields decoded lines.
        If the file does not exist yet, retries with back-off.
        """
        retry_delay = 1.0
        max_retry_delay = 30.0

        while True:
            if not os.path.exists(self.log_source):
                logger.warning(
                    "Log source '%s' not found (element %s). Retrying in %.1fs …",
                    self.log_source,
                    self.element_id,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue

            retry_delay = 1.0  # reset on success
            try:
                self._process = subprocess.Popen(
                    ["tail", "-F", "-n", "0", self.log_source],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                assert self._process.stdout is not None
                for line in self._process.stdout:
                    yield line
            except (subprocess.SubprocessError, OSError) as exc:
                logger.error(
                    "Tailing '%s' failed: %s. Restarting in %.1fs …",
                    self.log_source,
                    exc,
                    retry_delay,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            finally:
                if self._process and self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()

    def stop(self) -> None:
        """Stop the tail process gracefully."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()


# ---------------------------------------------------------------------------
# Batch file reader
# ---------------------------------------------------------------------------


class BatchLogReader:
    """Reads an existing log file line-by-line (for replay / debugging)."""

    def __init__(self, file_path: str, element_id: str) -> None:
        self.file_path = file_path
        self.element_id = element_id

    def read_lines(self) -> Generator[str, None, None]:
        """Yield lines from the file."""
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    yield line
        except FileNotFoundError:
            logger.error("Batch input file not found: %s", self.file_path)
        except OSError as exc:
            logger.error("Error reading batch file '%s': %s", self.file_path, exc)


# ---------------------------------------------------------------------------
# Kafka producer wrapper
# ---------------------------------------------------------------------------


class KafkaProducer:
    """
    Thin wrapper around ``kafka-python`` producer.
    Falls back to a mock/noop producer when the library is unavailable.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        client_id: str,
        security_protocol: str = "PLAINTEXT",
        batch_size: int = 16384,
        linger_ms: int = 100,
    ) -> None:
        self.topic = topic
        self.client_id = client_id
        self._producer: Optional[Any] = None
        self._closed = False

        try:
            from kafka import KafkaProducer as _KafkaProducer

            self._producer = _KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                client_id=client_id,
                security_protocol=security_protocol,
                key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                batch_size=batch_size,
                linger_ms=linger_ms,
                acks="all",
                retries=3,
                retry_backoff_ms=200,
                max_block_ms=10000,
                request_timeout_ms=15000,
            )
            logger.info(
                "Kafka producer connected to %s (topic=%s, client=%s)",
                bootstrap_servers,
                topic,
                client_id,
            )
        except ImportError:
            logger.warning(
                "kafka-python not installed; messages will be logged to stdout only."
            )
        except Exception as exc:
            logger.error("Failed to initialize Kafka producer: %s", exc)

    def produce(self, key: str, value: Dict[str, Any]) -> None:
        """Produce a single message to the configured topic."""
        if self._producer is not None:
            try:
                future = self._producer.send(self.topic, key=key, value=value)
                # Non-blocking — errors logged via callback
                future.add_errback(self._on_error)
            except Exception as exc:
                logger.error("Kafka produce error: %s", exc)
        else:
            # Fallback: log to stdout
            logger.debug(
                "[FALLBACK-PRODUCE] topic=%s key=%s value=%s",
                self.topic,
                key,
                json.dumps(value, default=str),
            )

    def produce_to_topic(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        """Produce a message to an arbitrary topic."""
        if self._producer is not None:
            try:
                future = self._producer.send(topic, key=key, value=value)
                future.add_errback(self._on_error)
            except Exception as exc:
                logger.error("Kafka produce error (topic=%s): %s", topic, exc)
        else:
            logger.debug(
                "[FALLBACK-PRODUCE] topic=%s key=%s value=%s",
                topic,
                key,
                json.dumps(value, default=str),
            )

    @staticmethod
    def _on_error(exc: Exception) -> None:
        """Callback for async produce errors."""
        logger.error("Kafka async produce error: %s", exc)

    def flush(self) -> None:
        """Flush pending messages."""
        if self._producer is not None:
            self._producer.flush(timeout=10)

    def close(self) -> None:
        """Close the producer."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
            if self._producer is not None:
                self._producer.close(timeout=5)
            logger.info("Kafka producer closed.")
        except Exception as exc:
            logger.error("Error closing Kafka producer: %s", exc)


# ---------------------------------------------------------------------------
# Telemetry Transformer
# ---------------------------------------------------------------------------


class TelemetryTransformer:
    """
    Transforms parsed SyslogEvent objects into TelemetryPoint objects
    compatible with the NetworkState proto schema consumed by the
    inference pipeline.
    """

    # Unit map for well-known metrics
    _UNIT_MAP: Dict[str, str] = {
        "rsrp": "dBm",
        "rsrq": "dB",
        "rssi": "dBm",
        "sinr": "dB",
        "tx_power": "dBm",
        "rx_power": "dBm",
        "dl_throughput_bps": "bps",
        "ul_throughput_bps": "bps",
        "arfcn": "channel",
        "pci": "id",
        "tac": "id",
        "qci": "class",
        "arp": "priority",
        "bearer_id": "id",
    }

    def __init__(self, element_id: str, element_label: str) -> None:
        self.element_id = element_id
        self.element_label = element_label

    def transform(self, event: SyslogEvent) -> List[TelemetryPoint]:
        """
        Transform a SyslogEvent into zero or more TelemetryPoints.
        Attempts radio, bearer, and event extraction from the message.
        """
        points: List[TelemetryPoint] = []
        ts = self._normalize_timestamp(event.timestamp)

        # 1) Radio metrics
        radio = self._estimate_radio_metrics(event.message)
        for key, val in radio.items():
            unit = self._UNIT_MAP.get(key, "unknown")
            points.append(
                TelemetryPoint(
                    element_id=self.element_id,
                    element_label=self.element_label,
                    metric_name=f"radio.{key}",
                    metric_value=float(val) if isinstance(val, (int, float)) else 0.0,
                    unit=unit,
                    timestamp=ts,
                    source="syslog.radio",
                    extra={
                        "severity": event.severity,
                        "source_host": event.source_host,
                    },
                )
            )

        # 2) Bearer metrics
        bearer = self._estimate_bearer_metrics(event.message)
        for key, val in bearer.items():
            unit = self._UNIT_MAP.get(key, "unknown")
            points.append(
                TelemetryPoint(
                    element_id=self.element_id,
                    element_label=self.element_label,
                    metric_name=f"bearer.{key}",
                    metric_value=float(val) if isinstance(val, (int, float)) else 0.0,
                    unit=unit,
                    timestamp=ts,
                    source="syslog.bearer",
                    extra={"severity": event.severity},
                )
            )

        # 3) Events (encode as telemetry with a boolean flag)
        evt = self._estimate_resource_metrics(event.message)
        if evt:
            points.append(
                TelemetryPoint(
                    element_id=self.element_id,
                    element_label=self.element_label,
                    metric_name=f"event.{evt.get('event_type', 'unknown')}",
                    metric_value=1.0,
                    unit="event",
                    timestamp=ts,
                    source="syslog.event",
                    extra=evt,
                )
            )

        return points

    def _estimate_radio_metrics(self, message: str) -> Dict[str, Any]:
        """Extract radio-level KPIs from the message."""
        parser = OsmocomSyslogParser(self.element_id, self.element_label)
        fields = parser.parse_radio_metrics(message)
        return fields if fields else {}

    def _estimate_bearer_metrics(self, message: str) -> Dict[str, Any]:
        """Extract bearer-level KPIs from the message."""
        parser = OsmocomSyslogParser(self.element_id, self.element_label)
        fields = parser.parse_bearer_metrics(message)
        return fields if fields else {}

    def _estimate_resource_metrics(self, message: str) -> Dict[str, Any]:
        """Extract event-level indicators from the message."""
        parser = OsmocomSyslogParser(self.element_id, self.element_label)
        fields = parser.parse_event(message)
        return fields if fields else {}

    @staticmethod
    def _normalize_timestamp(ts: str) -> str:
        """
        Normalize various syslog timestamp formats to ISO-8601 UTC.
        Handles: 'May 15 14:32:01', ISO-8601, epoch, etc.
        """
        now = datetime.now(timezone.utc)

        # RFC 3164: "May 15 14:32:01"
        try:
            parsed = datetime.strptime(ts, "%b %d %H:%M:%S")
            parsed = parsed.replace(year=now.year, tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            pass

        # ISO 8601-ish
        try:
            parsed = datetime.fromisoformat(ts)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except (ValueError, TypeError):
            pass

        # Fallback: current time
        return now.isoformat()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


class SyslogForwarderPipeline:
    """
    Orchestrates the full syslog → parse → transform → Kafka pipeline.
    Manages multiple log sources concurrently via threading.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._shutdown_event = False
        self._tailers: List[SyslogTailer] = []
        self._parsers: Dict[str, OsmocomSyslogParser] = {}
        self._transformers: Dict[str, TelemetryTransformer] = {}

        # Kafka producers
        kafka_cfg = config.get("kafka_config", {})
        self.producer_telemetry = KafkaProducer(
            bootstrap_servers=kafka_cfg.get("bootstrap_servers", "kafka:9092"),
            topic=kafka_cfg.get("topic_telemetry", "network-telemetry"),
            client_id=kafka_cfg.get("client_id", "syslog-forwarder"),
            security_protocol=kafka_cfg.get("security_protocol", "PLAINTEXT"),
        )
        self.producer_events = KafkaProducer(
            bootstrap_servers=kafka_cfg.get("bootstrap_servers", "kafka:9092"),
            topic=kafka_cfg.get("topic_events", "network-events"),
            client_id=f"{kafka_cfg.get('client_id', 'syslog-forwarder')}-events",
            security_protocol=kafka_cfg.get("security_protocol", "PLAINTEXT"),
        )

        # Parsing options
        self._parse_radio = config.get("parse_radio_metrics", True)
        self._parse_bearer = config.get("parse_bearer_metrics", True)
        self._parse_events = config.get("parse_events", True)

        # Batch accumulator
        self._batch_size = config.get("batch_size", 50)
        self._flush_interval = config.get("flush_interval_sec", 5)
        self._batch_buffer: deque = deque(maxlen=self._batch_size * 2)
        self._last_flush = time.monotonic()

        # Stats
        self._lines_processed = 0
        self._lines_skipped = 0
        self._points_produced = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the pipeline. Blocks until SIGTERM/SIGINT."""
        log_sources: List[LogSource] = []
        for src_cfg in self.config.get("log_sources", []):
            log_sources.append(
                LogSource(
                    path=src_cfg["path"],
                    element_id=src_cfg["element_id"],
                    element_label=src_cfg["element_label"],
                )
            )

        if not log_sources:
            logger.error("No log sources configured. Exiting.")
            return

        logger.info(
            "Starting Syslog Forwarder Pipeline with %d source(s)", len(log_sources)
        )

        # Create parsers and tailers for each source
        threads: List[threading.Thread] = []
        import threading

        for src in log_sources:
            parser = OsmocomSyslogParser(src.element_id, src.element_label)
            transformer = TelemetryTransformer(src.element_id, src.element_label)
            self._parsers[src.element_id] = parser
            self._transformers[src.element_id] = transformer

            tailer = SyslogTailer(src.path, src.element_id)
            self._tailers.append(tailer)

            t = threading.Thread(
                target=self._tail_loop,
                args=(tailer, src.element_id, parser, transformer),
                name=f"tailer-{src.element_id}",
                daemon=True,
            )
            threads.append(t)

        # Start all tailer threads
        for t in threads:
            t.start()
            logger.info("Started thread: %s", t.name)

        # Flush loop (runs on main thread)

        while not self._shutdown_event:
            elapsed = time.monotonic() - self._last_flush
            if self._batch_buffer and elapsed >= self._flush_interval:
                self._flush_batch()

            # Also check for batch-size trigger
            if len(self._batch_buffer) >= self._batch_size:
                self._flush_batch()

            time.sleep(0.1)

        # Shutdown: final flush
        logger.info(
            "Shutting down — flushing remaining %d messages …", len(self._batch_buffer)
        )
        self._flush_batch()
        self.stop()

    def _tail_loop(
        self,
        tailer: SyslogTailer,
        element_id: str,
        parser: OsmocomSyslogParser,
        transformer: TelemetryTransformer,
    ) -> None:
        """Inner loop for a single log source tailer thread."""
        try:
            for line in tailer.tail():
                if self._shutdown_event:
                    break
                self._process_line(line, element_id, parser, transformer)
        except Exception as exc:
            logger.error(
                "Tailer thread '%s' crashed: %s", element_id, exc, exc_info=True
            )
            self._errors += 1

    def _process_line(
        self,
        line: str,
        element_id: str,
        parser: OsmocomSyslogParser,
        transformer: TelemetryTransformer,
    ) -> None:
        """Parse a line, transform it, and buffer for Kafka production."""
        if self._shutdown_event:
            return

        self._lines_processed += 1

        # Parse syslog line
        event = parser.parse_line(line)
        if event is None:
            self._lines_skipped += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Unparseable line: %.120s", line.rstrip())
            return

        # Enrich with parsed fields
        event.parsed_fields = {}

        if self._parse_radio:
            radio = parser.parse_radio_metrics(event.message)
            if radio:
                event.parsed_fields["radio"] = radio

        if self._parse_bearer:
            bearer = parser.parse_bearer_metrics(event.message)
            if bearer:
                event.parsed_fields["bearer"] = bearer

        if self._parse_events:
            evt = parser.parse_event(event.message)
            if evt:
                event.parsed_fields["event"] = evt

        # Transform to telemetry points
        points = transformer.transform(event)

        for pt in points:
            self._batch_buffer.append(pt.to_dict())
            self._points_produced += 1

        # Periodic stats logging
        if self._lines_processed % 1000 == 0:
            logger.info(
                "Processed %d lines, %d skipped, %d telemetry points produced, %d errors",
                self._lines_processed,
                self._lines_skipped,
                self._points_produced,
                self._errors,
            )

    def _flush_batch(self) -> None:
        """Flush buffered telemetry points to Kafka."""
        if not self._batch_buffer:
            return

        batch = list(self._batch_buffer)
        self._batch_buffer.clear()
        self._last_flush = time.monotonic()

        for point_dict in batch:
            key = point_dict.get("element_id", "unknown")
            metric_name = point_dict.get("metric_name", "unknown")

            # Route events to the events topic
            if metric_name.startswith("event."):
                self.producer_events.produce(key=key, value=point_dict)
            else:
                self.producer_telemetry.produce(key=key, value=point_dict)

        logger.debug("Flushed %d telemetry points to Kafka", len(batch))

    def stop(self) -> None:
        """Graceful shutdown of all components."""
        self._shutdown_event = True

        # Stop tailers
        for tailer in self._tailers:
            tailer.stop()

        # Close Kafka producers
        self.producer_telemetry.close()
        self.producer_events.close()

        logger.info(
            "Pipeline stopped. Stats: lines=%d, skipped=%d, points=%d, errors=%d",
            self._lines_processed,
            self._lines_skipped,
            self._points_produced,
            self._errors,
        )


# ---------------------------------------------------------------------------
# Batch replay mode
# ---------------------------------------------------------------------------


def run_batch_mode(config: Dict[str, Any], input_file: str) -> None:
    """
    Read an existing log file and replay all lines through the
    parse → transform → Kafka pipeline without tailing.
    """
    logger.info("Running in BATCH mode — reading %s", input_file)

    # Use the first configured log source for element metadata
    sources = config.get("log_sources", [])
    element_id = sources[0]["element_id"] if sources else "BATCH-001"
    element_label = sources[0]["element_label"] if sources else "BatchReplay"

    parser = OsmocomSyslogParser(element_id, element_label)
    transformer = TelemetryTransformer(element_id, element_label)

    kafka_cfg = config.get("kafka_config", {})
    producer = KafkaProducer(
        bootstrap_servers=kafka_cfg.get("bootstrap_servers", "kafka:9092"),
        topic=kafka_cfg.get("topic_telemetry", "network-telemetry"),
        client_id=f"{kafka_cfg.get('client_id', 'syslog-forwarder')}-batch",
    )

    reader = BatchLogReader(input_file, element_id)
    lines = 0
    points = 0

    for line in reader.read_lines():
        event = parser.parse_line(line)
        if event is None:
            continue
        lines += 1

        for pt in transformer.transform(event):
            producer.produce(key=element_id, value=pt.to_dict())
            points += 1

    producer.flush()
    producer.close()
    logger.info("Batch complete: %d lines → %d telemetry points", lines, points)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def load_config(config_path: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration from:
      1. JSON config file (if provided)
      2. CLI arguments
      3. Environment variables
    CLI args take highest precedence, then env, then config file.
    """
    config: Dict[str, Any] = {}

    # 1) Load from file
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        logger.info("Loaded config from %s", config_path)

    # 2) Environment variable overrides
    env_overrides = {
        "kafka_config.bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        "kafka_config.topic_telemetry": os.getenv("KAFKA_TOPIC_TELEMETRY"),
        "kafka_config.topic_events": os.getenv("KAFKA_TOPIC_EVENTS"),
        "kafka_config.client_id": os.getenv("KAFKA_CLIENT_ID"),
        "kafka_config.security_protocol": os.getenv(
            "KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"
        ),
        "parse_radio_metrics": os.getenv("PARSE_RADIO_METRICS"),
        "parse_bearer_metrics": os.getenv("PARSE_BEARER_METRICS"),
        "parse_events": os.getenv("PARSE_EVENTS"),
        "batch_size": os.getenv("BATCH_SIZE"),
        "flush_interval_sec": os.getenv("FLUSH_INTERVAL_SEC"),
        "forwarder_mode": os.getenv("FORWARDER_MODE"),
    }

    for dotted_key, env_val in env_overrides.items():
        if env_val is None:
            continue
        # Navigate nested dict
        parts = dotted_key.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        raw_val = env_val
        # Boolean parsing
        if raw_val.lower() in ("true", "1", "yes"):
            raw_val = True  # type: ignore[assignment]
        elif raw_val.lower() in ("false", "0", "no"):
            raw_val = False  # type: ignore[assignment]
        # Int parsing
        elif raw_val.isdigit():
            raw_val = int(raw_val)  # type: ignore[assignment]
        target[parts[-1]] = raw_val

    # 3) CLI argument overrides
    if args.kafka_bootstrap:
        config.setdefault("kafka_config", {})["bootstrap_servers"] = (
            args.kafka_bootstrap
        )
    if args.kafka_topic:
        config.setdefault("kafka_config", {})["topic_telemetry"] = args.kafka_topic
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.flush_interval:
        config["flush_interval_sec"] = args.flush_interval

    # Ensure required structure
    config.setdefault("kafka_config", {})
    config["kafka_config"].setdefault("bootstrap_servers", "kafka:9092")
    config["kafka_config"].setdefault("topic_telemetry", "network-telemetry")
    config["kafka_config"].setdefault("topic_events", "network-events")
    config["kafka_config"].setdefault("client_id", "syslog-forwarder")
    config["kafka_config"].setdefault("security_protocol", "PLAINTEXT")
    config.setdefault("parse_radio_metrics", True)
    config.setdefault("parse_bearer_metrics", True)
    config.setdefault("parse_events", True)
    config.setdefault("batch_size", 50)
    config.setdefault("flush_interval_sec", 5)

    # Log sources from env (JSON array)
    if not config.get("log_sources"):
        log_sources_env = os.getenv("LOG_SOURCES")
        if log_sources_env:
            try:
                config["log_sources"] = json.loads(log_sources_env)
            except json.JSONDecodeError:
                logger.error("Failed to parse LOG_SOURCES env var as JSON")

    return config


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def setup_signal_handlers(pipeline: Optional[SyslogForwarderPipeline]) -> None:
    """Register SIGTERM and SIGINT handlers for graceful shutdown."""

    def _signal_handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown …", sig_name)
        if pipeline:
            pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="syslog_forwarder",
        description="6G Digital Immunity — Syslog Forwarder Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file.",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["tail", "batch"],
        default=None,
        help="Operating mode: 'tail' (live follow) or 'batch' (replay file).",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input log file for batch mode.",
    )
    p.add_argument(
        "--log-source",
        type=str,
        default=None,
        help="Single log file path to tail.",
    )
    p.add_argument(
        "--element-id",
        type=str,
        default=None,
        help="Element ID for the log source.",
    )
    p.add_argument(
        "--element-label",
        type=str,
        default=None,
        help="Human-readable element label.",
    )
    p.add_argument(
        "--kafka-bootstrap",
        type=str,
        default=None,
        help="Kafka bootstrap servers (comma-separated).",
    )
    p.add_argument(
        "--kafka-topic",
        type=str,
        default=None,
        help="Kafka topic for telemetry.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of messages to batch before flushing.",
    )
    p.add_argument(
        "--flush-interval",
        type=int,
        default=None,
        help="Flush interval in seconds.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Logging setup
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.StreamHandler(sys.stderr),  # errors also go to stderr
        ],
    )
    # Only send ERROR+ to stderr
    logging.getLogger().handlers[1].setLevel(logging.ERROR)

    logger.info("6G Digital Immunity — Syslog Forwarder Pipeline v1.0.0")

    # Load config
    config = load_config(args.config, args)

    # Single-source CLI overrides
    if args.log_source:
        config.setdefault("log_sources", []).insert(
            0,
            {
                "path": args.log_source,
                "element_id": args.element_id or "CLI-001",
                "element_label": args.element_label or "CLI-Source",
            },
        )

    if not config.get("log_sources"):
        logger.error(
            "No log sources configured. Use --config, --log-source, or LOG_SOURCES env."
        )
        sys.exit(1)

    mode = args.mode or config.get("forwarder_mode", "tail")
    logger.info("Mode: %s", mode)

    if mode == "batch":
        input_file = args.input
        if not input_file:
            # Use first log source path as batch input
            input_file = config["log_sources"][0]["path"]
        run_batch_mode(config, input_file)
    else:
        # Live tail mode
        pipeline = SyslogForwarderPipeline(config)
        setup_signal_handlers(pipeline)
        try:
            pipeline.run()
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            pipeline.stop()


if __name__ == "__main__":
    main()
