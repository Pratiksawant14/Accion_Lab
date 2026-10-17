# IMPLEMENTATION PLAN & MILESTONE ROADMAP
## Industry Project – Action Lab | Team 5B: Edge Inference & Hardware Integration
**Document Status:** LIVING RECORD & EXECUTION TRACKER  
**Last Updated:** 2026-10-17  
**Authoritative Reference:** [`PROJECT_SOURCE_OF_TRUTH.md`](file:///d:/Accion_Lab/PROJECT_SOURCE_OF_TRUTH.md)

---

## 1. Project Purpose & Core Architecture

Team 5B is responsible for bridging machine learning model training and edge hardware deployment. Our core architecture unites three pre-trained domain checkpoints into an edge-optimized system:
* **Agriculture:** `checkpoints/Agriculture_bestweight.pt` (YOLOv8n @ 512×512)
* **Ships:** `checkpoints/best.pt` (YOLO26n / YOLO11n @ 640×640)
* **Space Electronics:** `checkpoints/space_electronics.pt` (YOLO26n / YOLO11n @ 640×640)

> **Architectural Non-Negotiable (ADR-001):** Direct weight tensor merging is physically impossible due to network backbone divergences (`C2f` vs. `C3k2/C2PSA`). All operations route through the **Unified Inference Orchestrator** with domain routing and canonical taxonomy normalization.

---

## 2. Global Class Taxonomy Standard (ADR-002)

| Global Class ID | Canonical Class Name | Domain Source | Local Domain Index |
|:---:|:---|:---|:---:|
| **0** | `healthy_crop` | Agriculture | 0 |
| **1** | `dead_crop` | Agriculture | 1 |
| **2** | `cargo_ship` | Ships | 0 |
| **3** | `passenger_ship` | Ships | 1 |
| **4** | `open` | Space Electronics | 0 |
| **5** | `short` | Space Electronics | 1 |

---

## 3. Verified Artifacts Inventory

* **Checkpoints:** All 3 checkpoints verified in `checkpoints/` with SHA-256 hashes and safe introspection metadata.
* **Datasets:** Full synthetic datasets received and verified in `Synthline/`:
  * `agriculture robotics/agriculture_final_dataset`: 972 images, 878 labels (train/val/test).
  * `environmental defense/ship_final_dataset`: 883 images, 767 labels, 4,000-row quality CSV.
  * `space electronics/space_final_dataset`: 887 images, 829 labels.
  * **Total Data:** 2,742 images, 2,474 YOLO label files across all 3 domains.
* **Core Code Completed:**
  * `src/inference/schema.py`: Canonical schema, `BoundingBox`, `is_pseudo_box`, `to_telemetry_payload()`.
  * `src/inference/taxonomy.py`: Bi-directional taxonomy translation manager.
  * `src/inference/adapters/`: `BaseModelAdapter`, `AgricultureAdapter`, `ShipsAdapter`, `SpaceElectronicsAdapter`.
  * `src/inference/router.py`: `UnifiedInferenceRouter` with domain routing and lazy loading.
  * `tests/`: 11/11 unit tests passing (100%).

---

## 4. Master Milestone Roadmap & Status Tracker

| Milestone | Target Deliverable | Requirements Tracked | Status | Verification Criteria |
|:---:|---|:---:|:---:|---|
| **B0** | Environment & Scaffolding | N/A | **COMPLETED** | Git repo, Python 3.13, PyTorch, Ultralytics, checkpoints in place |
| **ADR-001** | Unified Inference Orchestrator | REQ-01, REQ-04 | **COMPLETED** | Adapters + Router dispatching live inference without cross-domain pollution |
| **B1** | Automated Dataset Pre-Flight Validator | REQ-02 | **COMPLETED** | Image integrity, bounds $[0, 1]$, class consistency, pseudo-box detection |
| **B6** | Performance Benchmarking Harness | REQ-11 | **COMPLETED** | $\ge 100$ frames per domain, FPS, p50/p95/p99 latency percentiles (`reports/benchmarks/`) |
| **B4** | ONNX Export Pipeline | REQ-05 | **COMPLETED** | Automated export script converting `.pt` checkpoints to `.onnx` (`artifacts/exports/`) |
| **B5** | Output Equivalence Testing | REQ-05 | **COMPLETED** | Max absolute difference $\le 10^{-3}$ confirmed across all 3 models (`reports/exports/`) |
| **B9** | Resilient MQTT Telemetry Client | REQ-09, REQ-10 | **COMPLETED** | 1,000-msg circular buffer, exp backoff reconnection, 60s failure alert (`src/telemetry/`) |
| **B8** | Edge Resource Constraint Simulation | REQ-08 | **COMPLETED** | Throttled CPU execution and memory budget enforcement (`src/simulation/`) |
| **B2** | Automated Training Pipeline Runner | REQ-01, REQ-03 | **COMPLETED** | YAML config-driven YOLO trainer supporting Nano/Small/Medium (`src/training/`) |
| **B3** | Evaluation & Resumption Suite | REQ-03, REQ-11 | **COMPLETED** | Automated mAP@0.5 calculation, checkpoint tracking, dataset pre-flight check gate |
| **B10** | Model Packaging & Registry Manifest | REQ-12 | **COMPLETED** | Auditable release manifests with SHA-256 hashes, Git commit, telemetry (`artifacts/releases/`) |
| **B11** | End-to-End Integrated Live Demo | REQ-13 | **COMPLETED** | Live multi-domain streaming demo to Team 5C telemetry specification (`scripts/demo_integrated_pipeline.py`) |

---

## 5. Step-by-Step Execution Plan

### Step 1: Dependencies Installation
* Install remaining requirements: `onnx>=1.14.0` and `paho-mqtt<2.0.0`.
* Verify environment readiness.

### Step 2: Milestone B1 — Dataset Pre-Flight & Sanity Validator (`src/data/validator.py`)
* Implement dataset scanner validating:
  1. Image readability and format integrity (JPEG, PNG).
  2. Bounding box coordinates in normalized $[0.0, 1.0]$ bounds.
  3. Label classes matching valid domain classes $\{0, 1\}$.
  4. Detection and flagging of synthetic pseudo-boxes (`width > 0.95 and height > 0.95`).
  5. Generating comprehensive summary reports (JSON/Markdown).
* Create automated unit tests in `tests/test_dataset_validator.py`.
* Run validator against `Synthline/` datasets.

### Step 3: Milestone B6 — Latency & FPS Performance Benchmarking (`src/benchmarking/`)
* Build benchmark harness running $\ge 100$ consecutive frames across Agriculture, Ships, and Space Electronics.
* Calculate:
  * FPS (Frames Per Second).
  * p50, p95, p99 latency percentiles (preprocessing, model forward pass, postprocessing, total).
  * Pseudo-box detection percentage per domain.
  * Verify edge publishing target ($\le 50$ms pure model inference).

### Step 4: Milestones B4 & B5 — ONNX Export & Output Equivalence Pipeline (`src/export/`)
* Build `src/export/onnx_exporter.py` with dynamic/fixed batch support.
* Implement numerical equivalence tester comparing PyTorch vs. ONNX Runtime on identical inputs ($|y_{pytorch} - y_{onnx}| \le 10^{-3}$).
* Run export and verification for all 3 domain checkpoints.

### Step 5: Milestones B9 & B8 — Resilient MQTT Telemetry & Edge Simulation (`src/telemetry/`, `src/simulation/`)
* Implement `src/telemetry/mqtt_publisher.py` with:
  * Thread-safe circular buffer (capacity $\ge 1,000$ messages).
  * Exponential backoff reconnection with jitter.
  * Alerting after $\ge 60$ seconds sustained network disconnection.
* Implement edge resource simulation utility (CPU process throttle, memory budgeting).
* Write chaos unit tests with disconnected mock broker.

### Step 6: Milestones B2 & B3 — Automated Training & Resumption Pipeline (`src/training/`)
* Implement standardized YAML configuration parser for model variant, batch size, learning rate, epochs.
* Implement training execution and checkpoint saving (`best.pt`, `last.pt`).
* Implement pre-flight validation gate ensuring dataset health before launching training.
* Implement ModelEvaluator computing mAP@0.5, mAP@0.5:0.95, and inference speeds.

### Step 7: Milestones B10 & B11 — Model Packaging & End-to-End Demo (`src/registry/`, `scripts/`)
* Implement packaging utility generating signed audit manifests (model SHA-256, dataset hash, Git commit, test metrics).
* Package Agriculture, Ships, and Space Electronics release bundles in `artifacts/releases/`.
* Implement integrated demo script simulating multi-domain live video feeds, real-time domain routing, canonical output formatting, and telemetry publication.

---

## 6. Execution Log & Changelog

| Timestamp (UTC) | Milestone | Action Taken | Result |
|---|:---:|---|---|
| 2026-10-11 | B0 / ADR-001 | Checkpoint introspection, schema, taxonomy, adapters, router created | 11/11 tests passed |
| 2026-10-17 | Discovery | `Synthline` dataset (2,742 images) audited and verified across all 3 domains | Confirmed class mapping & pseudo-boxes |
| 2026-10-17 | Roadmap | Initialized authoritative `docs/IMPLEMENTATION_PLAN.md` execution tracker | Document created |
| 2026-10-17 | Dependencies | Installed `onnx>=1.14.0` and `paho-mqtt<2.0.0` | All dependencies verified OK |
| 2026-10-17 | B1 (REQ-02) | Built `src/data/validator.py`, verified all 2,742 images & labels, wrote unit tests | 13/13 tests passed (100%), reports saved |
| 2026-10-17 | B6 (REQ-11) | Built benchmark harness, executed 100 frames/domain (300 total), profiled FPS/latency | 16/16 tests passed; reports saved in `reports/benchmarks/` |
| 2026-10-17 | B4, B5 (REQ-05) | Built ONNX exporter & numerical equivalence validator; exported all 3 models | 19/19 tests passed; max abs error <= 6.4e-4 <= 1e-3 (100% PASS) |
| 2026-10-17 | B9, B8 (REQ-09,10,08) | Built resilient MQTT publisher (1k buffer, exp backoff, 60s alert) & edge limiter | 26/26 tests passed (100%), chaos buffer test verified |
| 2026-10-17 | B2, B3 (REQ-01,03,04) | Built training config schema, runner with pre-flight check gate, and model evaluator | 34/34 tests passed (100%), YAML configs created |
| 2026-10-17 | B10 (REQ-12) | Built ModelPackager & release manifest generator; packaged all 3 models | 37/37 tests passed (100%), zip bundles generated in `artifacts/releases/` |
| 2026-10-17 | B11 (REQ-13) | Built `scripts/demo_integrated_pipeline.py`; executed multi-domain live streaming | 100% operational, real-time routing & telemetry verified (`reports/demos/`) |

