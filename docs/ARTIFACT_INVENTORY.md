# ARTIFACT INVENTORY & TRACKING RECORD
## Team 5B – Edge Inference & Hardware Integration
**Last Updated:** 2026-10-11  
**Workspace Root:** `d:/Accion_Lab`  

This inventory is verified against the actual filesystem on each inspection. Do not mark an artifact as present until its existence and checksum have been verified locally.

---

## 1. Summary of Inventory Status

| Category | Expected Count | Verified Present | Missing / Awaiting |
|---|:---:|:---:|:---:|
| Model Checkpoints (`.pt`) | 3 | **3** | 0 |
| Training Notebooks (`.ipynb`) | 3 | 0 | 3 |
| Dataset Configurations (`data.yaml`) | 3 | **3 (Embedded & Synthline)** | 0 |
| Training Configurations (`args.yaml`) | 3 | **3 (Embedded recovered)** | 0 |
| Training Metrics (`results.csv` / logs) | 3 | **1 (Ships quality validation)** | 2 |
| Evaluation Datasets / Samples | 3 | **3 (Full 2,742 images)** | 0 |
| Requirements & Interface Specs | 2 | 0 | 2 |
| **Total Core Artifacts** | **20** | **13** | **7** |

---

## 2. Model Checkpoints

| Checkpoint Name | Assigned Domain | Target Path | Status | File Size | SHA256 Hash | Verified Architecture & Classes |
|---|---|---|:---:|:---:|:---:|---|
| `Agriculture_bestweight.pt` | Agriculture | `checkpoints/Agriculture_bestweight.pt` | **VERIFIED PRESENT** | 5.94 MB (6,232,106 B) | `4480b9996668c8f7d3b4e62fc1456e45ebed01cc52f7b2072ea16470ac093b4f` | `yolov8n.pt` @ 512x512<br>`{0: 'healthy_crop', 1: 'dead_crop'}` |
| `best.pt` | Ships | `checkpoints/best.pt` | **VERIFIED PRESENT** | 5.14 MB (5,388,613 B) | `304f2f9b883473f1a307f5a976ee2afd0103d4cb0112a76995d28bc650f7c448` | `yolo26n.pt` @ 640x640<br>`{0: 'cargo_ship', 1: 'passenger_ship'}` |
| `space_electronics.pt` | Space Electronics | `checkpoints/space_electronics.pt` | **VERIFIED PRESENT** | 5.14 MB (5,388,549 B) | `50f7a4588d3f2996124cbe53f0ea7275a62db37313ce026d07d2c45b8a89645f` | `yolo26n.pt` @ 640x640<br>`{0: 'open', 1: 'short'}` |

---

## 3. Dataset Configuration & Metadata Files

| Artifact Name | Domain | Target Path | Status | Verification Findings |
|---|---|---|:---:|---|
| `data.yaml` (Agriculture) | Agriculture | `configs/datasets/agriculture_data.yaml` | **EMBEDDED EXTRACTED** | Embedded origin: `C:\Users\kjkar\Downloads\train\data.yaml`; Classes: `healthy_crop`, `dead_crop` |
| `data.yaml` (Ships) | Ships | `configs/datasets/ships_data.yaml` | **EMBEDDED EXTRACTED** | Embedded origin: `/content/ship_final_dataset/ship_final_dataset/data_local.yaml`; Classes: `cargo_ship`, `passenger_ship` |
| `data.yaml` (Space Electronics) | Space Electronics | `configs/datasets/space_electronics_data.yaml` | **EMBEDDED EXTRACTED** | Embedded origin: `/content/space_dataset/data.yaml`; Classes: `open`, `short` |

---

## 4. Training Notebooks & Configuration Records

| Artifact Name | Domain | Target Path | Status | Content / Model Version |
|---|---|---|:---:|---|
| Agriculture Training Notebook | Agriculture | `notebooks/agriculture_training.ipynb` | **AWAITING** | Needed for original augmentations & notebook audit |
| Ships Training Notebook | Ships | `notebooks/ships_training.ipynb` | **AWAITING** | Needed for original augmentations & notebook audit |
| Space Electronics Training Notebook | Space Electronics | `notebooks/space_electronics_training.ipynb` | **AWAITING** | Needed for original augmentations & notebook audit |
| Agriculture `args.yaml` | Agriculture | `configs/training/agriculture_args.yaml` | **RECOVERED FROM CKPT** | 50 epochs, batch 16, imgsz 512, lr0 0.01, lrf 0.01, patience 15, CPU |
| Ships `args.yaml` | Ships | `configs/training/ships_args.yaml` | **RECOVERED FROM CKPT** | 50 epochs, batch 16, imgsz 640, lr0 0.01, lrf 0.01, patience 100, GPU |
| Space Electronics `args.yaml` | Space Electronics | `configs/training/space_electronics_args.yaml` | **RECOVERED FROM CKPT** | 50 epochs, batch 16, imgsz 640, lr0 0.01, lrf 0.01, patience 100, GPU |

---

## 5. Training Results & Metrics

| Artifact Name | Domain | Target Path | Status | Baseline Reported Metrics |
|---|---|---|:---:|---|
| Agriculture Results | Agriculture | `reports/baselines/agriculture_results.csv` | **MISSING** | Baseline mAP / loss unverified |
| Ships Results | Ships | `reports/baselines/ships_results.csv` | **MISSING** | Baseline mAP / loss unverified |
| Space Electronics Results | Space Electronics | `reports/baselines/space_electronics_results.csv` | **MISSING** | Baseline mAP / loss unverified |

---

## 6. Datasets & Evaluation Samples

| Dataset Split | Domain | Target Path | Status | Verified Image Count | Verified Label Count |
|---|---|---|:---:|:---:|:---:|
| Full Splits (Train/Val/Test) | Agriculture | `Synthline/agriculture robotics/agriculture_final_dataset/` | **VERIFIED PRESENT** | 972 images (598 train, 245 val, 129 test) | 878 labels (632 train, 177 val, 69 test) |
| Full Splits (Train/Val/Test) | Ships | `Synthline/environmental defense/ship_final_dataset/` | **VERIFIED PRESENT** | 883 images (591 train, 213 val, 79 test) | 767 labels (489 train, 198 val, 80 test) |
| Full Splits (Train/Val/Test) | Space Electronics | `Synthline/space electronics/space_final_dataset/` | **VERIFIED PRESENT** | 887 images (590 train, 220 val, 77 test) | 829 labels (587 train, 166 val, 76 test) |
| Quality Validation CSV | Ships | `Synthline/environmental defense/ship_quality_validation.csv` | **VERIFIED PRESENT** | 4,000 records | PASS / REVIEW audit table |
| **Total Verified Data** | **All 3 Domains** | `Synthline/` | **VERIFIED PRESENT** | **2,742 images** | **2,474 YOLO label files** |

---

## 7. External Specifications & Interface Agreements

| Specification Document | Source | Target Path | Status | Verification Notes |
|---|---|---|:---:|---|
| Original Written Requirements Specification | Project Lead | `docs/specs/ORIGINAL_REQUIREMENTS.md` | **MISSING** | Reconstructed targets preserved in Source of Truth |
| MQTT JSON Telemetry Schema & Broker Spec | Team 5C | `docs/specs/TEAM_5C_MQTT_SCHEMA.json` | **MISSING** | Topic names, payload schemas, port/auth needed |
