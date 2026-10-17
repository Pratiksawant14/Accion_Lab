"""Automated Edge Vision Model Training Runner.

Integrates DatasetValidator pre-flight validation, handles model invocation
via Ultralytics YOLO API, manages checkpoint persistence, and tracks run artifacts.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.data.validator import DatasetValidator
from src.training.config import TrainingConfig


class TrainingResult(BaseModel):
    """Execution summary of a training run."""

    domain: str
    model_variant: str
    run_dir: str
    best_checkpoint_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    validation_passed: bool = True
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TrainingRunner:
    """Orchestrates end-to-end dataset validation and model training."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def run_pre_flight_check(self) -> bool:
        """Run dataset validator pre-flight check if configured."""
        if not self.config.validate_dataset_before_train:
            return True

        target_dir = self.config.dataset_root
        if not target_dir:
            # Attempt to derive root from data_yaml parent
            target_dir = str(Path(self.config.data_yaml).parent)

        if not Path(target_dir).exists():
            raise FileNotFoundError(
                f"Dataset directory '{target_dir}' does not exist for pre-flight check."
            )

        validator = DatasetValidator(domain=self.config.domain)
        report = validator.validate_dataset(dataset_root=target_dir)

        if not report.is_valid_for_training:
            corrupt_images = sum(s.corrupt_images for s in report.splits.values())
            corrupt_labels = sum(s.corrupt_labels for s in report.splits.values())
            invalid_boxes = sum(s.invalid_coordinate_boxes for s in report.splits.values())
            raise RuntimeError(
                f"Pre-flight validation failed for domain '{self.config.domain}': "
                f"{corrupt_images} corrupt images, {corrupt_labels} corrupt labels, "
                f"{invalid_boxes} invalid coordinate boxes. "
                "Resolve dataset errors before training."
            )

        return True

    def train(self, dry_run: bool = False) -> TrainingResult:
        """Execute the training job.

        Args:
            dry_run: If True, validates config, executes pre-flight checks,
                     and simulates checkpoint generation without invoking full training loop.

        Returns:
            TrainingResult with paths to best/last checkpoints and training statistics.
        """
        start_time = time.perf_counter()

        # 1. Pre-flight check gate
        validation_passed = self.run_pre_flight_check()

        run_dir = Path(self.config.project) / self.config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_path: Optional[Path] = None
        last_path: Optional[Path] = None
        metrics: Dict[str, Any] = {}

        if dry_run:
            # Simulate checkpoint creation for pipeline validation & testing
            best_path = weights_dir / "best.pt"
            last_path = weights_dir / "last.pt"
            best_path.write_text("MOCK_CHECKPOINT_BEST_WEIGHTS", encoding="utf-8")
            last_path.write_text("MOCK_CHECKPOINT_LAST_WEIGHTS", encoding="utf-8")
            metrics = {
                "dry_run": True,
                "epochs": self.config.epochs,
                "batch": self.config.batch,
                "imgsz": self.config.imgsz,
            }
        else:
            from ultralytics import YOLO

            model = YOLO(self.config.model_variant)
            args = self.config.to_ultralytics_args()
            results = model.train(**args)

            # Locate resulting checkpoints
            expected_best = weights_dir / "best.pt"
            expected_last = weights_dir / "last.pt"
            if expected_best.exists():
                best_path = expected_best
            if expected_last.exists():
                last_path = expected_last

            if hasattr(results, "results_dict"):
                metrics = dict(results.results_dict)

        duration = time.perf_counter() - start_time

        return TrainingResult(
            domain=self.config.domain,
            model_variant=self.config.model_variant,
            run_dir=str(run_dir.resolve()),
            best_checkpoint_path=str(best_path.resolve()) if best_path and best_path.exists() else None,
            last_checkpoint_path=str(last_path.resolve()) if last_path and last_path.exists() else None,
            validation_passed=validation_passed,
            duration_seconds=round(duration, 4),
            metrics=metrics,
        )
