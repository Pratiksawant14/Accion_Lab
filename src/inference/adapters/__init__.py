"""Domain Model Adapters for Team 5B.

Exposes domain-specific adapter classes configured with verified checkpoint paths,
architectures, and input resolutions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Type, Union

from .base import BaseModelAdapter
from .yolo_adapter import YOLOModelAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class AgricultureAdapter(YOLOModelAdapter):
    """Adapter for Agriculture domain model (YOLOv8n trained at 512x512)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = REPO_ROOT / "checkpoints" / "Agriculture_bestweight.pt",
        device: str = "cpu",
    ):
        super().__init__(
            domain="agriculture",
            checkpoint_path=checkpoint_path,
            target_imgsz=512,  # Verified from checkpoint training args
            model_name="agriculture_yolov8n",
            model_version="1.0.0",
            device=device,
        )


class ShipsAdapter(YOLOModelAdapter):
    """Adapter for Ships maritime domain model (YOLO26n trained at 640x640)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = REPO_ROOT / "checkpoints" / "best.pt",
        device: str = "cpu",
    ):
        super().__init__(
            domain="ships",
            checkpoint_path=checkpoint_path,
            target_imgsz=640,  # Verified from checkpoint training args
            model_name="ships_yolo26n",
            model_version="1.0.0",
            device=device,
        )


class SpaceElectronicsAdapter(YOLOModelAdapter):
    """Adapter for Space Electronics domain model (YOLO26n trained at 640x640)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = REPO_ROOT / "checkpoints" / "space_electronics.pt",
        device: str = "cpu",
    ):
        super().__init__(
            domain="space_electronics",
            checkpoint_path=checkpoint_path,
            target_imgsz=640,  # Verified from checkpoint training args
            model_name="space_electronics_yolo26n",
            model_version="1.0.0",
            device=device,
        )


ADAPTER_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
    "agriculture": AgricultureAdapter,
    "ships": ShipsAdapter,
    "space_electronics": SpaceElectronicsAdapter,
}


def get_adapter(domain: str, **kwargs) -> BaseModelAdapter:
    """Factory function returning the appropriate domain adapter."""
    domain_key = domain.lower()
    if domain_key not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown domain '{domain}'. Supported: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[domain_key](**kwargs)


__all__ = [
    "BaseModelAdapter",
    "YOLOModelAdapter",
    "AgricultureAdapter",
    "ShipsAdapter",
    "SpaceElectronicsAdapter",
    "get_adapter",
]
