# ARTIFACT INVENTORY & TRACKING RECORD
## Team 5B – Edge Inference & Hardware Integration
**Last Updated:** 2026-10-11  
**Workspace Root:** `d:/Accion_Lab`  

This inventory is verified against the actual filesystem on each inspection. Do not mark an artifact as present until its existence and checksum have been verified locally.

---

## 1. Summary of Inventory Status

| Category | Expected Count | Verified Present | Missing / Awaiting |
|---|:---:|:---:|:---:|
| Model Checkpoints (`.pt`) | 3 | 0 | 3 |
| Training Notebooks (`.ipynb`) | 3 | 0 | 3 |
| Dataset Configurations (`data.yaml`) | 3 | 0 | 3 |
| Training Configurations (`args.yaml`) | 3 | 0 | 3 |
| Training Metrics (`results.csv` / logs) | 3 | 0 | 3 |
| Evaluation Datasets / Samples | 3 | 0 | 3 |
| Requirements & Interface Specs | 2 | 0 | 2 |
| **Total Core Artifacts** | **20** | **0** | **20** |

---

## 2. Model Checkpoints

| Checkpoint Name | Assigned Domain | Target Path | Status | File Size | SHA256 Hash | Notes |
|---|---|---|:---:|:---:|:---:|---|
| `Agriculture_bestweight.pt` | Agriculture | `checkpoints/Agriculture_bestweight.pt` | **MISSING** | - | - | Expected classes: `healthy_crop`, `dead_crop` |
| `best (1).pt` | Ships | `checkpoints/best (1).pt` | **MISSING** | - | - | Expected classes: `cargo_ship`, `passenger_ship` |
| `space_electronics.pt` | Space Electronics | `checkpoints/space_electronics.pt` | **MISSING** | - | - | Expected classes: `open`, `short` |

---

## 3. Dataset Configuration & Metadata Files

| Artifact Name | Domain | Target Path | Status | Verification Findings |
|---|---|---|:---:|---|
| `data.yaml` (Agriculture) | Agriculture | `configs/datasets/agriculture_data.yaml` | **MISSING** | Local class ID ordering unverified |
| `data.yaml` (Ships) | Ships | `configs/datasets/ships_data.yaml` | **MISSING** | Local class ID ordering unverified |
| `data.yaml` (Space Electronics) | Space Electronics | `configs/datasets/space_electronics_data.yaml` | **MISSING** | Local class ID ordering unverified (`open` vs `short`) |

---

## 4. Training Notebooks & Configuration Records

| Artifact Name | Domain | Target Path | Status | Content / Model Version |
|---|---|---|:---:|---|
| Agriculture Training Notebook | Agriculture | `notebooks/agriculture_training.ipynb` | **MISSING** | Needed to verify architecture, augmentations, resolution |
| Ships Training Notebook | Ships | `notebooks/ships_training.ipynb` | **MISSING** | Needed to verify architecture, augmentations, resolution |
| Space Electronics Training Notebook | Space Electronics | `notebooks/space_electronics_training.ipynb` | **MISSING** | Needed to verify architecture, augmentations, resolution |
| Agriculture `args.yaml` | Agriculture | `configs/training/agriculture_args.yaml` | **MISSING** | Hyperparameters used during training |
| Ships `args.yaml` | Ships | `configs/training/ships_args.yaml` | **MISSING** | Hyperparameters used during training |
| Space Electronics `args.yaml` | Space Electronics | `configs/training/space_electronics_args.yaml` | **MISSING** | Hyperparameters used during training |

---

## 5. Training Results & Metrics

| Artifact Name | Domain | Target Path | Status | Baseline Reported Metrics |
|---|---|---|:---:|---|
| Agriculture Results | Agriculture | `reports/baselines/agriculture_results.csv` | **MISSING** | Baseline mAP / loss unverified |
| Ships Results | Ships | `reports/baselines/ships_results.csv` | **MISSING** | Baseline mAP / loss unverified |
| Space Electronics Results | Space Electronics | `reports/baselines/space_electronics_results.csv` | **MISSING** | Baseline mAP / loss unverified |

---

## 6. Datasets & Evaluation Samples

| Dataset Split | Domain | Target Path | Status | Verified Image Count |
|---|---|---|:---:|:---:|
| Validation Sample | Agriculture | `data/agriculture/val/` | **MISSING** | 0 |
| Validation Sample | Ships | `data/ships/val/` | **MISSING** | 0 |
| Validation Sample | Space Electronics | `data/space_electronics/val/` | **MISSING** | 0 |

---

## 7. External Specifications & Interface Agreements

| Specification Document | Source | Target Path | Status | Verification Notes |
|---|---|---|:---:|---|
| Original Written Requirements Specification | Project Lead | `docs/specs/ORIGINAL_REQUIREMENTS.md` | **MISSING** | Reconstructed targets preserved in Source of Truth |
| MQTT JSON Telemetry Schema & Broker Spec | Team 5C | `docs/specs/TEAM_5C_MQTT_SCHEMA.json` | **MISSING** | Topic names, payload schemas, port/auth needed |
