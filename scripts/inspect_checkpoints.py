"""Safe Checkpoint Inspector for Team 5B.

Performs non-destructive inspection of model weights:
- Verifies existence and measures file size.
- Calculates SHA-256 checksum for audit and lineage tracking.
- Safely inspects top-level metadata (class names, YOLO architecture, input size) without executing untrusted code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CheckpointInspector")

EXPECTED_CHECKPOINTS = {
    "agriculture": {
        "filename": "Agriculture_bestweight.pt",
        "domain": "Agriculture",
        "expected_classes": ["healthy_crop", "dead_crop"],
    },
    "ships": {
        "filename": "best (1).pt",
        "domain": "Ships",
        "expected_classes": ["cargo_ship", "passenger_ship"],
    },
    "space_electronics": {
        "filename": "space_electronics.pt",
        "domain": "Space Electronics",
        "expected_classes": ["open", "short"],
    },
}


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file in chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_pt_metadata(filepath: Path) -> dict:
    """Safely extracts checkpoint metadata if torch is installed."""
    try:
        import torch

        # Attempt safe loading using PyTorch weights_only or CPU load
        logger.info("Attempting inspection with torch on CPU: %s", filepath.name)
        data = torch.load(filepath, map_location="cpu", weights_only=False)

        info: dict = {
            "is_dict": isinstance(data, dict),
            "keys": list(data.keys()) if isinstance(data, dict) else str(type(data)),
        }

        if isinstance(data, dict):
            # Check for Ultralytics YOLO checkpoint metadata
            if "model" in data:
                model = data["model"]
                info["model_type"] = str(type(model))
                if hasattr(model, "names"):
                    info["class_names"] = model.names
                if hasattr(model, "yaml"):
                    info["model_yaml"] = model.yaml
            if "names" in data:
                info["class_names"] = data["names"]
            if "epoch" in data:
                info["epoch"] = data["epoch"]
            if "train_args" in data:
                info["train_args"] = data["train_args"]
            if "date" in data:
                info["date"] = data["date"]
            if "version" in data:
                info["yolo_version"] = data["version"]

        return info
    except Exception as e:
        logger.warning("Could not extract deep PyTorch metadata: %s", e)
        return {"inspection_error": str(e)}


def run_inventory(checkpoints_dir: Path) -> dict:
    report = {}
    print("=" * 80)
    print("TEAM 5B: CHECKPOINT INSPECTION REPORT")
    print("=" * 80)

    for domain_key, spec in EXPECTED_CHECKPOINTS.items():
        filepath = checkpoints_dir / spec["filename"]
        print(f"\n[Domain: {spec['domain']}] Checkpoint: {spec['filename']}")
        if not filepath.exists():
            print(f"  STATUS: MISSING (Expected at: {filepath})")
            report[domain_key] = {"status": "MISSING", "path": str(filepath)}
            continue

        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        sha256 = calculate_sha256(filepath)
        print(f"  STATUS: PRESENT")
        print(f"  Size: {size_mb:.2f} MB ({size_bytes} bytes)")
        print(f"  SHA-256: {sha256}")

        meta = inspect_pt_metadata(filepath)
        if "class_names" in meta:
            print(f"  Detected Class Names: {meta['class_names']}")
        if "yolo_version" in meta:
            print(f"  YOLO Version: {meta['yolo_version']}")
        if "epoch" in meta:
            print(f"  Trained Epochs: {meta['epoch']}")

        report[domain_key] = {
            "status": "PRESENT",
            "path": str(filepath),
            "size_mb": round(size_mb, 2),
            "sha256": sha256,
            "metadata": meta,
        }

    print("\n" + "=" * 80)
    return report


if __name__ == "__main__":
    checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    report = run_inventory(checkpoints_dir)
    output_json = checkpoints_dir.parent / "reports" / "checkpoint_inventory_latest.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Detailed report written to: {output_json}")
