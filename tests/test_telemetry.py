"""Unit and chaos tests for MQTT Telemetry Publisher (Milestone B9 / REQ-09, REQ-10)."""

import json
import time
import pytest

from src.inference.schema import BoundingBox, Detection, InferenceResult
from src.telemetry.mqtt_publisher import (
    BufferOverflowPolicy,
    ConnectionState,
    MQTTTelemetryPublisher,
    TelemetryConfig,
)


def test_telemetry_payload_schema_compliance():
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
    det = Detection(
        box=box,
        confidence=0.95,
        local_class_id=0,
        local_class_name="cargo_ship",
        global_class_id=2,
        global_class_name="cargo_ship",
    )
    result = InferenceResult(
        frame_id="frame_test_001",
        domain="ships",
        model_name="ships_yolo26n",
        model_version="1.0.0",
        inference_time_ms=45.2,
        detections=[det],
    )

    payload = result.to_telemetry_payload()
    assert payload["frame_id"] == "frame_test_001"
    assert payload["domain"] == "ships"
    assert payload["latency_ms"] == 45.2
    assert payload["global_class_ids"] == [2]
    assert payload["class_labels"] == ["cargo_ship"]
    assert payload["confidence_scores"] == [0.95]
    assert len(payload["bounding_boxes"]) == 1


def test_buffer_capacity_and_overflow_policy():
    """Chaos test: Publish 1,200 messages into a 1,000-capacity buffer while offline."""
    config = TelemetryConfig(
        broker_host="nonexistent.broker.local",
        buffer_capacity=1000,
        overflow_policy=BufferOverflowPolicy.DROP_OLDEST,
    )
    publisher = MQTTTelemetryPublisher(config=config)

    # Publish 1,200 messages while offline
    for i in range(1200):
        data = {"frame_id": f"frame_{i}", "val": i}
        success = publisher.publish(data)
        assert success is True

    metrics = publisher.get_metrics()
    assert metrics["current_buffer_size"] == 1000
    assert metrics["total_buffered"] == 1200
    assert metrics["total_dropped"] == 200
    assert metrics["connection_state"] == ConnectionState.DEGRADED_BUFFERING.value


def test_buffer_drop_newest_policy():
    config = TelemetryConfig(
        broker_host="nonexistent.broker.local",
        buffer_capacity=50,
        overflow_policy=BufferOverflowPolicy.DROP_NEWEST,
    )
    publisher = MQTTTelemetryPublisher(config=config)

    for i in range(50):
        assert publisher.publish({"id": i}) is True

    # 51st message must be rejected/dropped under DROP_NEWEST
    assert publisher.publish({"id": 50}) is False
    assert publisher.buffer_size == 50
    assert publisher.get_metrics()["total_dropped"] == 1


def test_sustained_outage_warning_trigger():
    warning_received = []

    def on_warning(duration: float):
        warning_received.append(duration)

    config = TelemetryConfig(
        failure_warning_threshold_sec=0.1,  # Fast trigger for test
    )
    publisher = MQTTTelemetryPublisher(config=config, on_outage_warning_callback=on_warning)

    # Fast forward disconnected start time to simulate > 0.1s outage
    publisher._disconnected_start_time = time.time() - 0.5
    publisher.publish({"test": 1})

    assert len(warning_received) == 1
    assert warning_received[0] >= 0.1
    assert publisher.get_metrics()["outage_warning_active"] is True
