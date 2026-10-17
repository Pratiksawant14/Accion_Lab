# Release Manifest: release_ships_v1.0.0
- **Domain:** `ships`
- **Architecture:** `YOLO26n / YOLO11n`
- **Release Version:** `1.0.0`
- **Git Commit:** `ace8ccfa4bbb4d84d1e0a770a05f967f92865be4`
- **Input Resolution:** `640x640`
- **Generated At:** `2026-10-17T18:10:13.192766+00:00`

## Model Artifact Integrity Checksums

| Artifact | Filename | Size (Bytes) | SHA-256 Checksum |
|---|---|:---:|---|
| PyTorch Checkpoint | `best.pt` | 5,388,613 | `304f2f9b883473f1a307f5a976ee2afd0103d4cb0112a76995d28bc650f7c448` |
| ONNX Export | `ships.onnx` | 9,747,864 | `3d5ad5d76fd175892bfed64baf9fb87ce48e53fdbbc11a363f567ff152178bc3` |

## Evaluation Metrics

- **map50:** `0.918`
- **map50_95:** `0.742`
- **precision:** `0.898`
- **recall:** `0.884`

## Edge Hardware Benchmark Metrics

- **fps:** `4.22`
- **p50_inference_ms:** `192.53`
- **p95_inference_ms:** `204.11`
- **peak_ram_mb:** `421.6`

## Dataset Lineage & Provenance

- **dataset_root:** `Synthline/environmental defense/ship_final_dataset`
- **split_counts:** `{'train': 591, 'val': 213, 'test': 79}`
- **total_images:** `883`
- **total_labels:** `767`