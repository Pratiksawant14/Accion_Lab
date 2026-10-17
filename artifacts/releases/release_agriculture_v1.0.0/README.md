# Release Manifest: release_agriculture_v1.0.0
- **Domain:** `agriculture`
- **Architecture:** `YOLOv8n`
- **Release Version:** `1.0.0`
- **Git Commit:** `ace8ccfa4bbb4d84d1e0a770a05f967f92865be4`
- **Input Resolution:** `512x512`
- **Generated At:** `2026-10-17T18:10:12.402678+00:00`

## Model Artifact Integrity Checksums

| Artifact | Filename | Size (Bytes) | SHA-256 Checksum |
|---|---|:---:|---|
| PyTorch Checkpoint | `Agriculture_bestweight.pt` | 6,232,106 | `4480b9996668c8f7d3b4e62fc1456e45ebed01cc52f7b2072ea16470ac093b4f` |
| ONNX Export | `agriculture.onnx` | 12,179,013 | `ad3cbcd68516ef279730ce7ba719be2d10f7a8d2054178998279d8f92e3e87f7` |

## Evaluation Metrics

- **map50:** `0.942`
- **map50_95:** `0.781`
- **precision:** `0.925`
- **recall:** `0.908`

## Edge Hardware Benchmark Metrics

- **fps:** `5.39`
- **p50_inference_ms:** `147.54`
- **p95_inference_ms:** `160.22`
- **peak_ram_mb:** `399.4`

## Dataset Lineage & Provenance

- **dataset_root:** `Synthline/agriculture robotics/agriculture_final_dataset`
- **split_counts:** `{'train': 598, 'val': 245, 'test': 129}`
- **total_images:** `972`
- **total_labels:** `878`