"""Standard YOLO Model Adapter for Team 5B.

Implements BaseModelAdapter for Ultralytics YOLO checkpoints (v8n, v11n/yolo26n).
Translates raw model detections into normalized coordinate dictionaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np

from .base import BaseModelAdapter

logger = logging.getLogger("YOLOAdapter")


class YOLOModelAdapter(BaseModelAdapter):
    """Adapter for Ultralytics YOLO models."""

    def __init__(
        self,
        domain: str,
        checkpoint_path: Union[str, Path],
        target_imgsz: int = 640,
        model_name: Optional[str] = None,
        model_version: str = "1.0.0",
        device: str = "cpu",
    ):
        super().__init__(
            domain=domain,
            checkpoint_path=checkpoint_path,
            target_imgsz=target_imgsz,
            model_name=model_name,
            model_version=model_version,
        )
        self.device = device

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}. "
                "Ensure trained weights are placed in checkpoints/"
            )

        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO weights for %s from: %s", self.domain, self.checkpoint_path)
            self._raw_model = YOLO(str(self.checkpoint_path))
            self._is_loaded = True
            logger.info("Successfully loaded model '%s' on %s", self.model_name, self.device)
        except ImportError as e:
            logger.error("Ultralytics is not installed. Please run: pip install ultralytics")
            raise RuntimeError(
                "Missing 'ultralytics' dependency required for PyTorch YOLO inference. "
                "Install it using: pip install ultralytics"
            ) from e
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise

    def _run_raw_inference(
        self,
        image_np: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> Tuple[List[dict], float]:
        if not self._is_loaded or self._raw_model is None:
            raise RuntimeError("Model is not loaded. Call load() before running inference.")

        start_time = time.perf_counter()
        results = self._raw_model.predict(
            source=image_np,
            imgsz=self.target_imgsz,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )
        forward_time_ms = (time.perf_counter() - start_time) * 1000.0

        raw_detections: List[dict] = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                # Extract normalized coordinates: xyxyn is (N, 4) in [0, 1]
                boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes_xyxyn, confs, classes):
                    raw_detections.append({
                        "x_min": float(box[0]),
                        "y_min": float(box[1]),
                        "x_max": float(box[2]),
                        "y_max": float(box[3]),
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                    })

                # If Ultralytics recorded precise internal inference speed, prefer that:
                if hasattr(result, "speed") and "inference" in result.speed:
                    forward_time_ms = float(result.speed["inference"])

        return raw_detections, forward_time_ms
