"""Automated Model Evaluator & Metrics Tracker.

Computes mAP@0.5, mAP@0.5:0.95, precision, recall, and inference speed
across validation or test splits, and outputs structured evaluation reports.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Structured evaluation metrics summary."""

    domain: str
    checkpoint_path: str
    data_yaml: str
    split: str
    map50: float = Field(ge=0.0, le=1.0)
    map50_95: float = Field(ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    fitness: float = Field(default=0.0)
    speed_ms: Dict[str, float] = Field(default_factory=dict)
    per_class_map50: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def save_report(self, output_path: str | Path) -> None:
        """Persist evaluation report as JSON."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


class ModelEvaluator:
    """Evaluates trained YOLO checkpoints on dataset splits."""

    def __init__(self, checkpoint_path: str | Path, domain: str = "agriculture"):
        self.checkpoint_path = Path(checkpoint_path)
        self.domain = domain
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

    def evaluate(
        self,
        data_yaml: str | Path,
        split: str = "val",
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cpu",
        mock_result: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate model against a dataset split.

        Args:
            data_yaml: Path to dataset YAML configuration.
            split: Split to evaluate ('val' or 'test').
            imgsz: Resolution for evaluation.
            batch: Batch size.
            device: Compute device ('cpu', 'cuda', etc.).
            mock_result: Optional mocked dictionary for fast unit testing.
        """
        data_path = Path(data_yaml)
        if not data_path.exists() and mock_result is None:
            raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

        if mock_result is not None:
            return EvaluationResult(
                domain=self.domain,
                checkpoint_path=str(self.checkpoint_path),
                data_yaml=str(data_path),
                split=split,
                map50=mock_result.get("map50", 0.85),
                map50_95=mock_result.get("map50_95", 0.65),
                precision=mock_result.get("precision", 0.88),
                recall=mock_result.get("recall", 0.82),
                fitness=mock_result.get("fitness", 0.70),
                speed_ms=mock_result.get("speed_ms", {"preprocess": 1.2, "inference": 150.0, "postprocess": 2.1}),
                per_class_map50=mock_result.get("per_class_map50", {}),
            )

        from ultralytics import YOLO

        model = YOLO(str(self.checkpoint_path))
        val_args = {
            "data": str(data_path),
            "split": split,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "verbose": False,
        }

        results = model.val(**val_args)

        # Extract metrics safely from Ultralytics results object
        map50 = float(results.box.map50) if hasattr(results.box, "map50") else 0.0
        map50_95 = float(results.box.map) if hasattr(results.box, "map") else 0.0
        precision = float(results.box.mp) if hasattr(results.box, "mp") else 0.0
        recall = float(results.box.mr) if hasattr(results.box, "mr") else 0.0
        fitness = float(results.fitness) if hasattr(results, "fitness") else 0.0
        speed = dict(results.speed) if hasattr(results, "speed") else {}

        per_class: Dict[str, float] = {}
        if hasattr(results.box, "maps") and results.box.maps is not None:
            names = results.names if hasattr(results, "names") else {}
            for i, val in enumerate(results.box.maps):
                name = names.get(i, f"class_{i}")
                per_class[str(name)] = float(val)

        return EvaluationResult(
            domain=self.domain,
            checkpoint_path=str(self.checkpoint_path),
            data_yaml=str(data_path),
            split=split,
            map50=round(map50, 4),
            map50_95=round(map50_95, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            fitness=round(fitness, 4),
            speed_ms={k: round(v, 2) for k, v in speed.items()},
            per_class_map50=per_class,
        )
