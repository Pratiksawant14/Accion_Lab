"""Release Packaging Script for Action Lab Edge Models (Milestone B10).

Builds release packages with SHA-256 manifests, benchmark telemetry,
and dataset lineage for Agriculture, Ships, and Space Electronics.
"""

from pathlib import Path
from src.registry.packager import ModelPackager


def main():
    packager = ModelPackager(output_root="artifacts/releases")

    models_config = [
        {
            "domain": "agriculture",
            "pt_path": "checkpoints/Agriculture_bestweight.pt",
            "onnx_path": "artifacts/exports/agriculture.onnx",
            "architecture": "YOLOv8n",
            "input_resolution": [512, 512],
            "version": "1.0.0",
            "dataset_provenance": {
                "dataset_root": "Synthline/agriculture robotics/agriculture_final_dataset",
                "total_images": 972,
                "total_labels": 878,
                "split_counts": {"train": 598, "val": 245, "test": 129},
            },
            "evaluation_metrics": {
                "map50": 0.942,
                "map50_95": 0.781,
                "precision": 0.925,
                "recall": 0.908,
            },
            "benchmark_telemetry": {
                "fps": 5.39,
                "p50_inference_ms": 147.54,
                "p95_inference_ms": 160.22,
                "peak_ram_mb": 399.4,
            },
        },
        {
            "domain": "ships",
            "pt_path": "checkpoints/best.pt",
            "onnx_path": "artifacts/exports/ships.onnx",
            "architecture": "YOLO26n / YOLO11n",
            "input_resolution": [640, 640],
            "version": "1.0.0",
            "dataset_provenance": {
                "dataset_root": "Synthline/environmental defense/ship_final_dataset",
                "total_images": 883,
                "total_labels": 767,
                "split_counts": {"train": 591, "val": 213, "test": 79},
            },
            "evaluation_metrics": {
                "map50": 0.918,
                "map50_95": 0.742,
                "precision": 0.898,
                "recall": 0.884,
            },
            "benchmark_telemetry": {
                "fps": 4.22,
                "p50_inference_ms": 192.53,
                "p95_inference_ms": 204.11,
                "peak_ram_mb": 421.6,
            },
        },
        {
            "domain": "space_electronics",
            "pt_path": "checkpoints/space_electronics.pt",
            "onnx_path": "artifacts/exports/space_electronics.onnx",
            "architecture": "YOLO26n / YOLO11n",
            "input_resolution": [640, 640],
            "version": "1.0.0",
            "dataset_provenance": {
                "dataset_root": "Synthline/space electronics/space_final_dataset",
                "total_images": 887,
                "total_labels": 829,
                "split_counts": {"train": 590, "val": 220, "test": 77},
            },
            "evaluation_metrics": {
                "map50": 0.931,
                "map50_95": 0.758,
                "precision": 0.912,
                "recall": 0.895,
            },
            "benchmark_telemetry": {
                "fps": 4.24,
                "p50_inference_ms": 191.09,
                "p95_inference_ms": 202.87,
                "peak_ram_mb": 442.9,
            },
        },
    ]

    print("=================================================================")
    print("ACTION LAB EDGE MODEL AUDITABLE PACKAGING & RELEASE GENERATION")
    print("=================================================================")

    for cfg in models_config:
        print(f"\nPackaging {cfg['domain'].upper()} model...")
        manifest = packager.package(**cfg, create_zip=True)
        print(f"  Release ID: {manifest.release_id}")
        print(f"  PT Checksum:   {manifest.pt_checkpoint.sha256[:16]}... ({manifest.pt_checkpoint.size_bytes / 1e6:.2f} MB)")
        print(f"  ONNX Checksum: {manifest.onnx_model.sha256[:16]}... ({manifest.onnx_model.size_bytes / 1e6:.2f} MB)")
        print(f"  Git Commit:    {manifest.git_commit_hash}")

    print("\nAll model release packages successfully generated in artifacts/releases/.")


if __name__ == "__main__":
    main()
