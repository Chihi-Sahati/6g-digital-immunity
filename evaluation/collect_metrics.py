import os
import json
import time
import logging
from confluent_kafka import Consumer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("metrics-collector")


def collect_metrics(duration_sec=60, output_file="/app/metrics_log.json"):
    kafka_brokers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    logger.info(f"Connecting to Kafka at {kafka_brokers} to collect metrics...")

    conf = {
        "bootstrap.servers": kafka_brokers,
        "group.id": "metrics-collector-group",
        "auto.offset.reset": "latest",
    }
    consumer = Consumer(conf)
    consumer.subscribe(["network-telemetry", "safety-audit-log"])

    start_time = time.time()
    metrics = {"telemetry": [], "audit_logs": []}

    logger.info(f"Listening for {duration_sec} seconds...")

    try:
        while time.time() - start_time < duration_sec:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue

            topic = msg.topic()
            try:
                data = json.loads(msg.value().decode("utf-8"))
                if topic == "network-telemetry":
                    metrics["telemetry"].append(data)
                elif topic == "safety-audit-log":
                    metrics["audit_logs"].append(data)
                    logger.info(
                        f"Received audit log: {data.get('decision', 'UNKNOWN')}"
                    )
            except Exception:
                pass

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user.")
    finally:
        consumer.close()

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(
        f"Collected {len(metrics['telemetry'])} telemetry points and {len(metrics['audit_logs'])} audit logs."
    )
    logger.info(f"Saved to {output_file}")


if __name__ == "__main__":
    collect_metrics()
