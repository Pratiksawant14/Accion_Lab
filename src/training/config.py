"""Standardized Training Configuration Schema and Loader.

Supports YOLO Nano/Small/Medium architecture variants, hyperparameter
configuration, pre-flight dataset validation toggles, and checkpoint tracking.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """Schema for edge vision model training."""

    domain: str = Field(
        ...,
        description="Target domain: 'agriculture', 'ships', or 'space_electronics'",
    )
    model_variant: str = Field(
        default="yolov8n.pt",
        description="Base architecture weights or YAML (e.g. 'yolov8n.pt', 'yolo11n.pt')",
    )
    data_yaml: str = Field(
        ...,
        description="Path to dataset YAML file (Ultralytics format)",
    )
    dataset_root: Optional[str] = Field(
        default=None,
        description="Root directory of dataset for pre-flight validation check",
    )
    imgsz: int = Field(default=640, ge=64, le=1280)
    epochs: int = Field(default=50, ge=1, le=1000)
    batch: int = Field(default=16, ge=1, le=256)
    lr0: float = Field(default=0.01, gt=0.0, lt=1.0)
    optimizer: str = Field(default="auto")
    device: str = Field(default="cpu")
    workers: int = Field(default=2, ge=0)
    patience: int = Field(default=15, ge=1)
    save_period: int = Field(default=10, ge=1)
    project: str = Field(default="runs/train")
    name: str = Field(default="exp")
    resume: bool = Field(default=False)
    validate_dataset_before_train: bool = Field(
        default=True,
        description="If True, executes DatasetValidator pre-flight check before training.",
    )

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        valid_domains = {"agriculture", "ships", "space_electronics"}
        val = v.lower().strip()
        if val not in valid_domains:
            raise ValueError(
                f"Invalid domain '{v}'. Must be one of {valid_domains}"
            )
        return val

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML content in {file_path}")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)

    def to_ultralytics_args(self) -> Dict[str, Any]:
        """Convert config to dictionary format suitable for YOLO.train()."""
        return {
            "data": self.data_yaml,
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "lr0": self.lr0,
            "optimizer": self.optimizer,
            "device": self.device,
            "workers": self.workers,
            "patience": self.patience,
            "save_period": self.save_period,
            "project": self.project,
            "name": self.name,
            "resume": self.resume,
        }
