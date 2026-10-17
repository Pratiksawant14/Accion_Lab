# Release Manifest: release_space_electronics_v1.0.0
- **Domain:** `space_electronics`
- **Architecture:** `YOLO26n / YOLO11n`
- **Release Version:** `1.0.0`
- **Git Commit:** `ace8ccfa4bbb4d84d1e0a770a05f967f92865be4`
- **Input Resolution:** `640x640`
- **Generated At:** `2026-10-17T18:10:13.916080+00:00`

## Model Artifact Integrity Checksums

| Artifact | Filename | Size (Bytes) | SHA-256 Checksum |
|---|---|:---:|---|
| PyTorch Checkpoint | `space_electronics.pt` | 5,388,549 | `50f7a4588d3f2996124cbe53f0ea7275a62db37313ce026d07d2c45b8a89645f` |
| ONNX Export | `space_electronics.onnx` | 9,747,819 | `6082629db480fe1933dca033ffb8617eff5716c8ae337fe39a5bdd8af11de6af` |

## Evaluation Metrics

- **map50:** `0.931`
- **map50_95:** `0.758`
- **precision:** `0.912`
- **recall:** `0.895`

## Edge Hardware Benchmark Metrics

- **fps:** `4.24`
- **p50_inference_ms:** `191.09`
- **p95_inference_ms:** `202.87`
- **peak_ram_mb:** `442.9`

## Dataset Lineage & Provenance

- **dataset_root:** `Synthline/space electronics/space_final_dataset`
- **split_counts:** `{'train': 590, 'val': 220, 'test': 77}`
- **total_images:** `887`
- **total_labels:** `829`