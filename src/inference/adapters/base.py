"""Base Model Adapter for Team 5B Unified Inference Pipeline.

Defines the common abstract interface that all domain-specific model adapters
(Agriculture, Ships, Space Electronics) must inherit and implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ..schema import BoundingBox, Detection, InferenceRequest, InferenceResult
from ..taxonomy import taxonomy_manager

logger = logging.getLogger("ModelAdapter")


class BaseModelAdapter(ABC):
    """Abstract base class for all domain model adapters."""

    def __init__(
        self,
        domain: str,
        checkpoint_path: Union[str, Path],
        target_imgsz: int = 640,
        model_name: Optional[str] = None,
        model_version: str = "1.0.0",
    ):
        self.domain = domain.lower()
        self.checkpoint_path = Path(checkpoint_path)
        self.target_imgsz = target_imgsz
        self.model_name = model_name or f"{self.domain}_detector"
        self.model_version = model_version
        self._is_loaded = False
        self._raw_model = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """Loads model weights into memory on target device."""
        pass

    @abstractmethod
    def _run_raw_inference(
        self,
        image_np: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> Tuple[List[dict], float]:
        """Executes raw forward pass and postprocessing.

        Returns:
            raw_detections: list of dicts with keys:
                - x_min, y_min, x_max, y_max (normalized in [0, 1])
                - confidence (float)
                - class_id (local int)
            forward_time_ms: pure model latency in milliseconds
        """
        pass

    def _prepare_image(self, image_input: Any) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Loads and normalizes image input into a numpy array (H, W, C) RGB.

        Returns:
            image_np: RGB numpy array
            orig_shape: (orig_width, orig_height)
            prep_time_ms: preprocessing duration
        """
        start_time = time.perf_counter()

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at: {path}")
            pil_img = Image.open(path).convert("RGB")
            orig_shape = pil_img.size  # (W, H)
            image_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            orig_shape = pil_img.size
            image_np = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            orig_shape = (image_input.shape[1], image_input.shape[0])  # (W, H)
            image_np = image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        prep_time_ms = (time.perf_counter() - start_time) * 1000.0
        return image_np, orig_shape, prep_time_ms

    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Executes full standardized inference pipeline for this domain model."""
        if not self._is_loaded:
            logger.info("Model '%s' not loaded yet. Auto-loading...", self.model_name)
            self.load()

        # 1. Preprocess
        image_np, (orig_w, orig_h), prep_time_ms = self._prepare_image(request.image)

        # 2. Forward pass
        raw_detections, forward_time_ms = self._run_raw_inference(
            image_np=image_np,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
        )

        # 3. Postprocess and normalize to canonical taxonomy
        post_start = time.perf_counter()
        normalized_detections: List[Detection] = []

        for raw in raw_detections:
            local_id = int(raw["class_id"])
            local_name = taxonomy_manager.get_local_name(self.domain, local_id)
            global_id, global_name = taxonomy_manager.to_global(self.domain, local_id)

            box = BoundingBox(
                x_min=max(0.0, min(1.0, float(raw["x_min"]))),
                y_min=max(0.0, min(1.0, float(raw["y_min"]))),
                x_max=max(0.0, min(1.0, float(raw["x_max"]))),
                y_max=max(0.0, min(1.0, float(raw["y_max"]))),
            )

            detection = Detection(
                box=box,
                confidence=round(float(raw["confidence"]), 4),
                local_class_id=local_id,
                local_class_name=local_name,
                global_class_id=global_id,
                global_class_name=global_name,
            )
            normalized_detections.append(detection)

        post_time_ms = (time.perf_counter() - post_start) * 1000.0

        frame_id = request.frame_id or f"frame_{int(time.time() * 1000)}"

        return InferenceResult(
            frame_id=frame_id,
            domain=self.domain,
            model_name=self.model_name,
            model_version=self.model_version,
            inference_time_ms=forward_time_ms,
            preprocess_time_ms=prep_time_ms,
            postprocess_time_ms=post_time_ms,
            detections=normalized_detections,
            metadata={
                "target_imgsz": self.target_imgsz,
                "orig_dimensions": [orig_w, orig_h],
                "detections_count": len(normalized_detections),
            },
        )
