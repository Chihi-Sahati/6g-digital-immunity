import sys
import grpc
from google.protobuf import timestamp_pb2

# Update Python path so it can import the proto definitions inside the container
sys.path.insert(0, "/app/api/proto")

import llm_intent_pb2 as intent_pb2
import llm_intent_pb2_grpc as intent_pb2_grpc


def test_intent():
    # Connect to the DSF server's internal address (no TLS for internal in our modified config)
    print("Connecting to DSF Server on dsf-server:50052...")
    channel = grpc.insecure_channel("dsf-server:50052")
    stub = intent_pb2_grpc.IntentValidationServiceStub(channel)

    # Construct a dummy intent
    ts = timestamp_pb2.Timestamp()
    ts.GetCurrentTime()

    intent = intent_pb2.NetworkIntent(
        intent_id="test-intent-001",
        rationale="Automated test to verify CBF safety filter bounds",
        generated_at=ts,
        target_element_id="bts-001",
        category=intent_pb2.IntentCategory.CAPACITY_ADJUSTMENT,
        confidence_score=0.95,
        radio=intent_pb2.RadioIntent(
            tx_power_dbm=43.0,  # Safe: bound is 0 to 46
            ho_hysteresis_db=3.0,  # Safe: bound is 0 to 30
        ),
    )

    print("Submitting intent to DSF Server for validation...")
    try:
        verdict = stub.ValidateIntent(intent, timeout=5.0)
        print("\n=== VERDICT RECEIVED ===")
        print(f"Decision: {intent_pb2.VerdictDecision.Name(verdict.decision)}")
        print(f"Verdict ID: {verdict.verdict_id}")

        if verdict.cbf_result:
            print("\n--- CBF Safety Check Results ---")
            print(f"Passed: {verdict.cbf_result.passed}")
            print(f"Min Margin: {verdict.cbf_result.minimum_margin}")

    except grpc.RpcError as e:
        print(f"\n[ERROR] gRPC Request failed: {e.details()} (Status: {e.code()})")


if __name__ == "__main__":
    test_intent()
