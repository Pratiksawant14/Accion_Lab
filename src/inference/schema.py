"""Canonical Inference Schemas for Team 5B Unified Inference Pipeline.

Defines standardized Pydantic models for detection outputs, bounding boxes,
inference requests, and edge telemetry-compatible payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class Domain(str, Enum):
    AGRICULTURE = "agriculture"
    SHIPS = "ships"
    SPACE_ELECTRONICS = "space_electronics"


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0] range."""

    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized top-left x")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized top-left y")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom-right x")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom-right y")

    @field_validator("x_max")
    @classmethod
    def validate_x_max(cls, v: float, info) -> float:
        if "x_min" in info.data and v < info.data["x_min"]:
            raise ValueError(f"x_max ({v}) must be >= x_min ({info.data['x_min']})")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y_max(cls, v: float, info) -> float:
        if "y_min" in info.data and v < info.data["y_min"]:
            raise ValueError(f"y_max ({v}) must be >= y_min ({info.data['y_min']})")
        return v

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def is_pseudo_box(self) -> bool:
        """Flags potential full-image pseudo-bounding boxes (e.g. >95% coverage)."""
        return self.width > 0.95 and self.height > 0.95


class Detection(BaseModel):
    """A single normalized detection with both local and global taxonomy mapping."""

    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    local_class_id: int = Field(..., ge=0, description="Domain model internal class index")
    local_class_name: str = Field(..., description="Domain model internal class label")
    global_class_id: int = Field(..., ge=0, le=5, description="Global system class index (0-5)")
    global_class_name: str = Field(..., description="Canonical global class name")


class InferenceRequest(BaseModel):
    """Unified inference request payload."""

    image: Any = Field(..., description="Image input: file path str, Path, numpy array, or PIL Image")
    domain: Optional[Domain] = Field(None, description="Target domain. If None, auto-route or error.")
    frame_id: Optional[str] = Field(None, description="Optional tracking frame identifier")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence cutoff")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="NMS IoU cutoff threshold")


class InferenceResult(BaseModel):
    """Standardized canonical inference result returned by router and ready for telemetry."""

    frame_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    domain: str
    model_name: str
    model_version: str
    inference_time_ms: float = Field(..., description="Pure model forward pass latency in milliseconds")
    preprocess_time_ms: float = Field(0.0, description="Image transformation latency in milliseconds")
    postprocess_time_ms: float = Field(0.0, description="NMS and coordinate transformation latency in milliseconds")
    detections: List[Detection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_telemetry_payload(self) -> Dict[str, Any]:
        """Formats output according to the agreed Team 5C edge telemetry specification."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "domain": self.domain,
            "model_version": self.model_version,
            "latency_ms": round(self.inference_time_ms, 2),
            "bounding_boxes": [
                [round(d.box.x_min, 4), round(d.box.y_min, 4), round(d.box.x_max, 4), round(d.box.y_max, 4)]
                for d in self.detections
            ],
            "class_labels": [d.global_class_name for d in self.detections],
            "global_class_ids": [d.global_class_id for d in self.detections],
            "confidence_scores": [round(d.confidence, 4) for d in self.detections],
        }
