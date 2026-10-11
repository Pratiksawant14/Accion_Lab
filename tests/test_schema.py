"""Unit Tests for Canonical Inference and Telemetry Schemas."""

import pytest
from pydantic import ValidationError
from src.inference.schema import BoundingBox, Detection, InferenceResult


def test_bounding_box_valid():
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
    assert box.width == pytest.approx(0.4)
    assert box.height == pytest.approx(0.4)
    assert not box.is_pseudo_box


def test_bounding_box_pseudo_box_detection():
    # Box covering almost the entire frame (>95%)
    box = BoundingBox(x_min=0.01, y_min=0.01, x_max=0.99, y_max=0.99)
    assert box.is_pseudo_box


def test_bounding_box_invalid_coords():
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.6, y_min=0.2, x_max=0.4, y_max=0.6)  # x_max < x_min

    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.1, y_min=0.7, x_max=0.4, y_max=0.3)  # y_max < y_min


def test_telemetry_payload_serialization():
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
    detection = Detection(
        box=box,
        confidence=0.925,
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
        inference_time_ms=18.45,
        preprocess_time_ms=2.1,
        postprocess_time_ms=1.3,
        detections=[detection],
    )

    payload = result.to_telemetry_payload()
    assert payload["frame_id"] == "frame_test_001"
    assert payload["domain"] == "ships"
    assert payload["latency_ms"] == 18.45
    assert len(payload["bounding_boxes"]) == 1
    assert payload["bounding_boxes"][0] == [0.1, 0.2, 0.5, 0.6]
    assert payload["class_labels"] == ["cargo_ship"]
    assert payload["global_class_ids"] == [2]
    assert payload["confidence_scores"] == [0.925]
