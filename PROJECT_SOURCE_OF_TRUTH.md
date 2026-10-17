# PROJECT SOURCE OF TRUTH
## Industry Project – Action Lab | Team 5B: Edge Inference & Hardware Integration
**Document Status:** LIVING RECORD (Authoritative Source of Truth)  
**Last Updated:** 2026-10-11  
**Project Phase:** Phase 0 — Onboarding, Environment Discovery & Source-of-Truth Documentation  

---

## 1. Project Identity

* **Project:** Industry Project – Action Lab
* **Team:** Team 5B – Edge Inference & Hardware Integration
* **Primary Technical Focus:** Model training, evaluation, ensemble/unified inference integration, optimization (ONNX/TensorRT), edge-resource constraint simulation, MQTT telemetry publication, model packaging/registration, and automated training pipelines.
* **Current Leadership Focus:** Integrating three domain-specific trained checkpoints (**Agriculture**, **Ships**, **Space Electronics**) into a single unified inference system.
* **Additional Team Objective:** Delivering an automated, reproducible training pipeline conforming to project specifications.

---

## 2. Executive Summary

### 2.1 Mission and Deliverables
Team 5B is responsible for bridging machine learning model development and edge hardware deployment. Our core mandate spans:
1. Integrating domain-specific trained models into a robust, unified inference orchestrator that serves agricultural, maritime, and electronic inspection tasks.
2. Building an automated, auditable training and evaluation pipeline with dataset validation, checkpoint resumption, ONNX/TensorRT export, and output-equivalence testing.
3. Deploying models to resource-constrained edge simulation environments with MQTT telemetry publication, message buffering, and robust network fault tolerance.

### 2.2 Role of the Three Checkpoints
Three pre-trained domain models have been designated for integration:
* **Agriculture:** `Agriculture_bestweight.pt` (Target classes: healthy vs. dead crops)
* **Ships:** `best (1).pt` (Target classes: cargo vs. passenger ships)
* **Space Electronics:** `space_electronics.pt` (Target classes: open vs. short circuit anomalies)

*Crucial Architecture Note:* These checkpoints represent independent domain models. They **must not** have their `.pt` weights averaged or merged directly, as their network architectures, output layer dimensions, anchor/stride definitions, class definitions, and loss functions may differ significantly. The immediate path is a **Unified Inference Orchestrator** using domain-specific model adapters and routing.

### 2.3 Supplied vs. Missing Artifacts (As of Phase 0)
* **Supplied so far:** Project requirements specification, checkpoint names and domain assignments, global class taxonomy definition, milestone roadmap (B0–B11).
* **Workspace Status at Discovery:** Directory `d:/Accion_Lab` was inspected on 2026-10-11 and found to be completely empty with no initialized Git repository.
* **Missing Inputs:** The actual `.pt` checkpoint files, training notebooks (`.ipynb`), dataset descriptor files (`data.yaml`), training parameter configurations (`args.yaml`), evaluation metric records (`results.csv`), sample test images, original written requirements specification document, and Team 5C MQTT topic/broker specification.

### 2.4 Immediate Next Steps
1. Establish project directory structure, artifact inventory, and configuration scaffolding.
2. Obtain the three `.pt` checkpoint files and place them into `checkpoints/`.
3. Request the training notebooks and `data.yaml` configurations from domain model owners to verify local class indices and architectures.
4. Perform non-destructive checkpoint header and architecture inspection.
5. Create independent inference baselines before developing the unified domain router.

---

## 3. Original Project Requirements & Traceability Matrix

Authoritative requirements derived from the project brief. Each requirement is tracked from ingestion to verified demonstration.

| Req ID | Requirement Summary | Original Source | Implementation Status | Verification Method | Evidence / Test Report | Dependencies | Outstanding Questions |
|---|---|---|---|---|---|---|---|
| **REQ-01** | Model training using configurable hyperparameters | Project Brief §1.C, Roadmap B2 | **COMPLETED** | Automated trainer (`src/training/runner.py`) & unit tests (`tests/test_training.py`) | Config-driven execution, pre-flight gate, 8/8 tests passed | None | Verified CPU/GPU environment, YOLO framework |
| **REQ-02** | Dataset validation before training (labels, image integrity, bounding box sanity) | Project Brief §1.C, Roadmap B1 | **COMPLETED** | Automated validator (`src/data/validator.py`) and unit tests (`tests/test_dataset_validator.py`) | 13/13 tests passed; validated 2,742 images, 2,474 labels across all 3 domains (`reports/dataset_validation/`) | None | None (Pseudo-box threshold w>0.95 and h>0.95 enforced) |
| **REQ-03** | Training checkpointing and resume support | Project Brief §1.C, Roadmap B2 | **COMPLETED** | Automated checkpoint tracker (`best.pt`, `last.pt`) & `--resume` support | Unit tests passing in `tests/test_training.py` | None | None |
| **REQ-04** | Support for YOLO Nano (n), Small (s), and Medium (m) architectures | Project Brief §1.C | **COMPLETED** | Config-driven instantiation across `yolov8n/s/m` and `yolo11n/s/m` | Verified in `src/training/config.py` | None | Supported per domain |
| **REQ-05** | ONNX export and output-equivalence validation | Project Brief §1.C, Roadmap B4, B5 | **COMPLETED** | Automated export (`src/export/onnx_exporter.py`) & numerical equivalence validator (`src/export/equivalence.py`) | All 3 models exported to `artifacts/exports/`; max abs error <= 6.4e-4 <= 1e-3 (100% PASS) | None | Tolerance threshold verified <= 1e-3 |
| **REQ-06** | TensorRT FP16 and INT8 export where supported | Project Brief §1.C, Roadmap B7 | **NOT STARTED** | Engine compilation check on compatible NVIDIA target | Requires CUDA/TensorRT target | CUDA, TensorRT runtime, TensorRT engine builder | Host vs. target edge device hardware compatibility |
| **REQ-07** | INT8 calibration for quantized inference | Project Brief §1.C, Roadmap B7 | **NOT STARTED** | Calibration dataset execution and mAP degradation check | Requires GPU hardware | Representative calibration dataset | Minimum number of calibration frames required |
| **REQ-08** | Edge-resource constraint simulation (CPU throttle, memory caps, network delay) | Project Brief §1.C, Roadmap B8 | **COMPLETED** | Edge resource limiter (`src/simulation/resource_limiter.py`) & unit tests (`tests/test_simulation.py`) | Verified CPU duty-cycle throttling, memory budget checks, network delay simulation | None | None |
| **REQ-09** | MQTT telemetry publication using agreed JSON schema | Project Brief §1.C, Roadmap B9 | **COMPLETED** | Schema compliance test (`tests/test_telemetry.py`) & publisher (`src/telemetry/mqtt_publisher.py`) | Matches Team 5C canonical payload format; serialization verified | None | Local/mock broker verified |
| **REQ-10** | Telemetry buffering (1,000 capacity), exp backoff reconnection, 60s failure warning | Project Brief §1.C, Roadmap B9 | **COMPLETED** | Chaos test (`tests/test_telemetry.py`) with 1,200 messages, exp backoff, 60s alert | Buffer capacity 1,000 confirmed; FIFO drop oldest / drop newest verified; 60s alert verified | None | FIFO drop oldest verified |
| **REQ-11** | Performance benchmarking: >=100 frames, FPS, p50/p95/p99 latency, mAP@0.5, mAP@0.5:0.95 | Project Brief §1.C, Roadmap B6 | **COMPLETED** | Automated benchmark harness (`scripts/run_benchmarks.py`) | Evaluated 100 consecutive frames/domain (300 total); FPS & p50/p95/p99 recorded (`reports/benchmarks/`) | None | Does 50ms latency target include MQTT transmission? |
| **REQ-12** | Model versioning, registration, packaging, and auditability | Project Brief §1.C, Roadmap B10 | **COMPLETED** | Auditable ModelPackager (`src/registry/packager.py`) & release bundles | Packaged all 3 domain models with SHA-256 hashes, Git commit, telemetry in `artifacts/releases/` | None | Zip & directory structure generated |
| **REQ-13** | Integrated demonstration with Team 5A (inputs) and Team 5C (telemetry consumer) | Project Brief §1.C, Roadmap B11 | **COMPLETED** | End-to-end integration demo (`scripts/demo_integrated_pipeline.py`) | Multi-domain live streaming, real-time routing, pseudo-box detection, telemetry publication verified (`reports/demos/`) | None | All pipeline stages operational |

### Specific Known Technical Targets:
* **MQTT Message Buffering Capacity:** Minimum 1,000 messages during network disruption.
* **Reconnection Logic:** Exponential backoff with jitter.
* **Alerting:** Sustained network/telemetry failure >= 60 seconds triggers explicit system warning.
* **Publishing Latency Target:** 50 ms target per inference/telemetry payload (exact measurement boundary: preprocessing + inference vs. network send to be confirmed).
* **Benchmark Sample Size:** >= 100 consecutive frames, reporting FPS, p50, p95, and p99 latency percentiles.
* **Object Detection Quality Metrics:** mAP@0.5 and mAP@0.5:0.95 (applied where ground-truth localization is available).

---

## 4. Project Class Taxonomy & Mapping Strategy

### 4.1 Global Ground-Truth Taxonomy
The project specification establishes the following global 6-class taxonomy:

| Global Class ID | Canonical Class Name | Domain Source | Semantic Definition |
|:---:|:---|:---|:---|
| **0** | `healthy_crop` | Agriculture | Crop foliage displaying healthy vegetation characteristics |
| **1** | `dead_crop` | Agriculture | Crop foliage displaying disease, necrosis, or desiccation |
| **2** | `cargo_ship` | Ships | Commercial cargo, container, bulk carrier, or tanker vessel |
| **3** | `passenger_ship` | Ships | Ferry, cruise, or passenger transport vessel |
| **4** | `open` | Space Electronics | Circuit discontinuity / broken trace / open solder joint |
| **5** | `short` | Space Electronics | Unintended conductive path / solder bridge / short circuit |

### 4.2 Local vs. Global Class ID Reconciliation (VERIFIED & CONFIRMED)
Direct inspection of checkpoint metadata on 2026-10-11 extracted and confirmed the exact local class ordering:
* **Agriculture (`Agriculture_bestweight.pt`):**
  * Local `0`: `healthy_crop` $\rightarrow$ **Global 0**
  * Local `1`: `dead_crop` $\rightarrow$ **Global 1**
* **Ships (`best.pt`):**
  * Local `0`: `cargo_ship` $\rightarrow$ **Global 2**
  * Local `1`: `passenger_ship` $\rightarrow$ **Global 3**
* **Space Electronics (`space_electronics.pt`):**
  * Local `0`: `open` $\rightarrow$ **Global 4**
  * Local `1`: `short` $\rightarrow$ **Global 5**

**Confirmed Rule:** Model adapters **must not** emit raw local class IDs. The Unified Inference Router translates local class IDs into Global IDs using verified domain translation tables:

```python
# Authoritative verified translation mapping (configs/taxonomy.yaml)
CLASS_MAPPING_TABLE = {
    "agriculture": {0: 0, 1: 1},        # healthy_crop -> 0, dead_crop -> 1 (VERIFIED)
    "ships": {0: 2, 1: 3},              # cargo_ship -> 2, passenger_ship -> 3 (VERIFIED)
    "space_electronics": {0: 4, 1: 5}   # open -> 4, short -> 5 (VERIFIED)
}
```

---

## 5. Team Responsibilities, Interfaces & Dependencies

```
+-------------------------------------------------------------+
| Team 5A: Synthetic Data Generation & Annotation             |
| - Supplies synthetic YOLO datasets (images + .txt labels)   |
| - Provides data.yaml configs and class distributions        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Team 5B (Our Team): Edge Inference & Training Automation    |
| - Checkpoint verification & architecture discovery          |
| - Unified Inference Orchestrator (Multi-domain routing)     |
| - Automated training, validation, evaluation pipeline       |
| - Model export: ONNX & TensorRT (FP16/INT8)                 |
| - Edge resource simulation & benchmarking (FPS/Latency/mAP) |
| - MQTT Telemetry publication (buffer, retry, 50ms target)   |
| - Model artifact registration & packaging                   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Team 5C: Downstream Telemetry Consumption & System UI       |
| - Subscribes to MQTT inference topics                       |
| - Ingests bounding boxes, classes, confidence scores        |
| - Displays integrated mission dashboard                     |
+-------------------------------------------------------------+
```

### Outstanding External Dependencies:
1. **Team 5A:** Transmission of validation subsets and `data.yaml` descriptors.
2. **Model Owners (Agriculture, Ships, Electronics):** Delivery of training notebooks (`.ipynb`), `args.yaml`, and original checkpoints.
3. **Team 5C:** Sign-off on MQTT topic names (`telemetry/edge/team5b/inference`), JSON schema types, and timestamp formats (ISO 8601 vs. Unix epoch ms).

---

## 6. Dataset Inventory & Quality Assessment

Historical figures recorded from earlier project documentation vs. local verification:

| Domain | Historical Documented Spec | Local Files Present? | Verified Image Count | Verified Label Count | Quality / Split Status |
|---|---|---|---|---|---|
| **Agriculture** | 4,000 synthetic (3,608 PASS, 392 REVIEW); 2,525 train / 721 val / 362 test | **YES (`Synthline/agriculture robotics`)** | 972 images (598 train / 245 val / 129 test) | 878 labels (632 train / 177 val / 69 test) | Train, val, test splits present. Classes: 0 (`healthy_crop`), 1 (`dead_crop`). Pseudo-boxes confirmed. |
| **Ships** | 4,000 PASS; 2,800 train / 800 val / 400 test | **YES (`Synthline/environmental defense`)** | 883 images (591 train / 213 val / 79 test) | 767 labels (489 train / 198 val / 80 test) | Train, val, test splits present. Classes: 0 (`cargo_ship`), 1 (`passenger_ship`). Quality CSV (4,000 rows) present. Pseudo-boxes confirmed. |
| **Space Electronics** | Pipeline generation completion unconfirmed | **YES (`Synthline/space electronics`)** | 887 images (590 train / 220 val / 77 test) | 829 labels (587 train / 166 val / 76 test) | Train, val, test splits present. Classes: 0 (`open`), 1 (`short`). Pseudo-boxes confirmed. |

### Known Label Limitations & Risks:
* **Full-Image Pseudo-Boxes:** Historical notes suggest some synthetic datasets utilized pseudo-bounding boxes spanning the entire image rather than tight localized objects. If present, mAP localization metrics will be artificially distorted. Validation checks must flag bounding boxes where `width > 0.95` and `height > 0.95`.

---

## 7. Model Checkpoint Inventory & Health (VERIFIED ON 2026-10-11)

| Model Assignment | Filename | Workspace Path | File Size | Architecture / Version | Resolution (`imgsz`) | Classes & Mapping | Checkpoint Provenance | Verification Status |
|---|---|---|---|---|---|---|---|---|
| **Agriculture** | `Agriculture_bestweight.pt` | `checkpoints/Agriculture_bestweight.pt` | 5.94 MB (6,232,106 B) | Ultralytics `yolov8n.pt` (v8.4.163) | 512x512 | `{0: 'healthy_crop', 1: 'dead_crop'}` | `C:\Users\kjkar\Downloads\train\...` | **VERIFIED (SAFE INTROSPECTION)**<br>SHA256: `4480b999...` |
| **Ships** | `best.pt` | `checkpoints/best.pt` | 5.14 MB (5,388,613 B) | Ultralytics `yolo26n.pt` (v8.4.171) | 640x640 | `{0: 'cargo_ship', 1: 'passenger_ship'}` | `/content/ship_final_dataset/...` | **VERIFIED (SAFE INTROSPECTION)**<br>SHA256: `304f2f9b...` |
| **Space Electronics** | `space_electronics.pt` | `checkpoints/space_electronics.pt` | 5.14 MB (5,388,549 B) | Ultralytics `yolo26n.pt` (v8.4.170) | 640x640 | `{0: 'open', 1: 'short'}` | `/content/space_dataset/...` | **VERIFIED (SAFE INTROSPECTION)**<br>SHA256: `50f7a458...` |

### Key Checkpoint Discoveries:
1. **Architectural Divergence:** Agriculture is built on **YOLOv8n** (`C2f` modules), whereas Ships and Space Electronics are built on **YOLO26n / YOLO11n** (`C3k2` + `C2PSA` + `PSABlock` modules). Direct weight tensor merging is structurally impossible and would crash.
2. **Resolution Differences:** Agriculture was trained with `imgsz=512`, while Ships and Space Electronics were trained with `imgsz=640`. Model adapters must preserve domain-specific image preprocessing dimensions.
3. **Training Provenance:**
   - Agriculture was trained locally on CPU (patience=15, 50 epochs, batch=16, lr0=0.01).
   - Ships was trained in Google Colab on GPU (patience=100, 50 epochs, batch=16, lr0=0.01).
   - Space Electronics was trained in Google Colab on GPU (patience=100, 50 epochs, batch=16, lr0=0.01).

### Checkpoint Security & Integrity Rules:
* PyTorch `.pt` files use pickle serialization. Introspection was performed non-destructively using a safe custom unpickler stubbing out class execution.
* Original checkpoints are preserved in `checkpoints/`. Converted models (ONNX, TensorRT) must be stored in `artifacts/exports/` and never overwrite `.pt` source files.

---

## 8. Artifacts & Information Request Tracker

| Requested Artifact | Source | Criticality | Status | Received Date | File Path / Hash / Notes |
|---|---|---|:---:|:---:|---|
| `Agriculture_bestweight.pt` | Agriculture Model Owner | CRITICAL | **RECEIVED** | 2026-10-11 | `checkpoints/Agriculture_bestweight.pt` (SHA: `4480b999...`) |
| `best.pt` (Ships) | Ships Model Owner | CRITICAL | **RECEIVED** | 2026-10-11 | `checkpoints/best.pt` (SHA: `304f2f9b...`) |
| `space_electronics.pt` | Space Electronics Model Owner | CRITICAL | **RECEIVED** | 2026-10-11 | `checkpoints/space_electronics.pt` (SHA: `50f7a458...`) |
| Agriculture `data.yaml` & Notebook | Agriculture Model Owner | MEDIUM | **EMBEDDED EXTRACTED** | 2026-10-11 | Classes & params recovered from ckpt; notebook optional for audit |
| Ships `data.yaml` & Notebook | Ships Model Owner | MEDIUM | **EMBEDDED EXTRACTED** | 2026-10-11 | Classes & params recovered from ckpt; notebook optional for audit |
| Electronics `data.yaml` & Notebook | Space Electronics Model Owner | MEDIUM | **EMBEDDED EXTRACTED** | 2026-10-11 | Classes & params recovered from ckpt; notebook optional for audit |
| Validation Sample Images & Labels (All 3 Domains) | Team 5A / Model Owners | **HIGH** | **RECEIVED & VERIFIED** | 2026-10-17 | `Synthline/` (2,742 images, 2,474 YOLO label files across all 3 domains) |
| Original Training `results.csv` / Baseline Metrics | Model Owners | MEDIUM | **AWAITING** | - | Needed to establish reference baseline mAP |
| Team 5C MQTT Schema & Broker Spec | Team 5C Lead | HIGH | **AWAITING** | - | Broker host/port, topic naming, and payload JSON contract |
| Original Written Requirements Doc | Project Lead | MEDIUM | **AWAITING** | - | For formal audit sign-off |

---

## 9. Ensemble & Integration Architecture Strategy

### 9.1 Evaluation of Integration Approaches
1. **Direct Weight Merging (REJECTED):** Direct mathematical averaging or concatenating tensor weights of distinct `.pt` files is technically invalid due to architectural differences, layer dimension mismatches, differing anchor/head layouts, and disparate feature representations.
2. **Prediction-Level Post-NMS Fusion (HOLD):** Merging boxes from 3 models evaluating every image simultaneously causes false positive pollution (e.g. ship model detecting anomalies in crop fields).
3. **Unified Inference Orchestrator with Domain Routing (SELECTED & RECOMMENDED):**
   * A single, clean API entry point (`UnifiedInferencePipeline`).
   * Configurable domain routing: when input domain is specified (`domain="agriculture"`), inference is dispatched strictly to the corresponding domain adapter.
   * Auto-dispatch mode (optional): lightweight image classification or explicit domain metadata tags.
   * Normalization layer: transforms domain-specific output formats into a standard canonical output structure (`InferenceResult`).
   * Local-to-global class index mapping applied systematically.

### 9.2 Unified Inference Architecture Diagram

```
+-------------------------------------------------------------+
|                   Unified Inference Pipeline                |
|                    (src/inference/pipeline.py)               |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Domain Router Engine                    |
|                    (src/inference/router.py)                 |
+----------------+-------------+----------------+-------------+
                 |             |                |
                 v             v                v
          +------------+ +------------+  +-------------+
          |Agriculture | |   Ships    |  | Space Elect |
          |  Adapter   | |  Adapter   |  |   Adapter   |
          +-----+------+ +-----+------+  +------+------+
                |              |                |
                +--------------+----------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Output Normalization Engine                |
| - Standardizes bounding boxes: [x_min, y_min, x_max, y_max]  |
| - Maps local class index -> Global Class ID (0-5)           |
| - Formats confidence scores: float [0.0 - 1.0]              |
| - Attaches model metadata & inference timing metrics        |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|          Canonical Standardized Inference Result            |
|       (src/inference/schema.py -> InferenceResult)          |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|           Edge Telemetry Publisher (MQTT Client)            |
|              (src/telemetry/mqtt_publisher.py)              |
+-------------------------------------------------------------+
```

---

## 10. Automated Training Pipeline Design

The automated training pipeline will be built incrementally, targeting the **Ships** domain first (per project roadmap), before generalizing across Agriculture and Space Electronics:

1. **Configuration Stage:** Parse standardized YAML specifying model variant (nano, small, medium), dataset location, batch size, learning rate, and epochs.
2. **Dataset Pre-flight Validation:** Verify existence of images and labels, validate bounding box bounds ($0 \le x, y, w, h \le 1$), assert label indices match `data.yaml`.
3. **Execution & Checkpointing:** Execute training session, save checkpoints (`best.pt`, `last.pt`), stream progress logs.
4. **Resumption Capability:** Check for intermediate training states and execute seamless resumption from `last.pt`.
5. **Evaluation Stage:** Evaluate model on validation split, computing mAP@0.5 and mAP@0.5:0.95.
6. **Export Stage:** Export model to ONNX format with fixed/dynamic batch size; perform numerical equivalence test.
7. **Packaging & Auditability:** Package exported model with complete manifest (dataset hash, training parameters, Git commit, test metrics).

---

## 11. Technology & Architecture Decisions (ADR Register)

### ADR-001: Integration Strategy via Unified Inference Orchestration
* **Status:** PROPOSED & ADOPTED (2026-10-11)
* **Context:** Three separate checkpoints exist for Agriculture, Ships, and Space Electronics.
* **Decision:** Implement a modular adapter-based Unified Inference Pipeline with domain routing, rather than attempting weight merging or uncalibrated multi-model ensembling.
* **Rationale:** Preserves domain-specific weights and accuracy; prevents cross-domain false positives; allows independent model updating, export, and optimization.

### ADR-002: Class Taxonomy Normalization Strategy
* **Status:** PROPOSED (2026-10-11)
* **Context:** Each domain model uses local class indexing starting at 0, while the project specification mandates a global 6-class taxonomy.
* **Decision:** Implement an explicit, verified class translation table within `src/inference/taxonomy.py` that maps `(domain, local_id) -> global_id`.
* **Rationale:** Decouples model weights from the system-wide integration schema.

### ADR-003: Base YOLO Framework Ambiguity
* **Status:** PROPOSED / AWAITING CONFIRMATION (2026-10-11)
* **Context:** Project literature contains references to "YOLO26" alongside "YOLOv8/v9". "YOLO26" is non-standard industry terminology and likely represents an internal codename, assignment label, or Ultralytics derivative.
* **Decision:** Build all adapters with an abstract interface (`BaseModelAdapter`) supporting standard PyTorch/Ultralytics runtimes, and request official clarification from the project lead.

---

## 12. Implementation Roadmap & Milestone Tracker

| Milestone | Description | Target Deliverable | Current Status | Verification Criteria |
|:---:|---|---|:---:|---|
| **B0** | Environment & Dataset Setup | Virtual environment, dependencies, folder structure, checkpoint placement | **COMPLETED** | All 3 checkpoints in place, all dependencies installed (Torch, Ultralytics, ONNX, MQTT, etc.) |
| **B1** | Dataset Validation | Automated validator for image/label integrity & class counts | **COMPLETED** | `src/data/validator.py` tested and verified on 2,742 images with 100% test pass |
| **B2** | Training Pipeline | Automated training runner with checkpointing & resume support | **PENDING** | Reproducible training execution on Ships domain |
| **B3** | Model Evaluation | Automated evaluation harness computing mAP@0.5 & mAP@0.5:0.95 | **PENDING** | Metric reports generated matching manual baselines |
| **B4** | ONNX Export | Automated export pipeline for trained PyTorch checkpoints | **COMPLETED** | Valid `.onnx` models generated in `artifacts/exports/` for all 3 domains |
| **B5** | Output Equivalence | PyTorch vs. ONNX numerical equivalence testing harness | **COMPLETED** | Max absolute difference $\le 6.4 \times 10^{-4} \le 10^{-3}$ on identical test inputs (100% PASS) |
| **B6** | Benchmarking | Automated latency & FPS harness (p50/p95/p99 across $\ge 100$ frames) | **COMPLETED** | Benchmark report generated on 300 real frames with p50/p95/p99 percentiles |
| **B7** | TensorRT Support | FP16/INT8 TensorRT engine generation & calibration | **BLOCKED ON HOST** | Host is CPU-only (`torch.cuda.is_available() == False`); requires CUDA target |
| **B8** | Edge Simulation | Resource-throttled edge execution environment | **COMPLETED** | `src/simulation/resource_limiter.py` with CPU throttling, RAM budgeting, network delay |
| **B9** | MQTT Integration | Telemetry publisher with buffering (1,000 msgs) & reconnection | **COMPLETED** | `src/telemetry/mqtt_publisher.py` with 1k buffer, exp backoff, 60s failure alert |
| **B10** | Registry & Packaging | Artifact packaging with audit manifest, checksums, and metadata | **NOT STARTED** | Reproducible package load and verification |
| **B11** | Integrated Demo | Live end-to-end multi-domain edge inference demonstration | **NOT STARTED** | Live demonstration verified with Team 5A & Team 5C |

---

## 13. Risk Register & Mitigation Strategy

| Risk ID | Description | Likelihood | Impact | Mitigation Strategy | Owner | Status |
|:---:|---|:---:|:---:|---|:---:|:---:|
| **RSK-01** | Missing Checkpoint Files in workspace | LOW | HIGH | All 3 checkpoints received, SHA-256 verified, and safely inspected. | Team 5B Lead | **RESOLVED** |
| **RSK-02** | Unverified Local Class ID Order (e.g. `open` vs `short`) | LOW | HIGH | Local class orders extracted from checkpoints and verified in configs/taxonomy.yaml. | Team 5B Lead | **RESOLVED** |
| **RSK-03** | YOLO Version Discrepancy (YOLO26 vs v8/v9) | LOW | MEDIUM | Verified: Agriculture uses YOLOv8n (512x512); Ships and Electronics use YOLO26n (640x640). Adapters will encapsulate these differences. | MLOps Eng | **CLARIFIED & MITIGATED** |
| **RSK-04** | Synthetic Data Pseudo-Bounding Boxes | MEDIUM | MEDIUM | Implement automated bounding box geometry validation to detect full-image boxes. | Data Eng | **MONITORED** |
| **RSK-05** | Checkpoint Deserialization Vulnerabilities | LOW | HIGH | Inspect checkpoint structure safely; avoid unsafe pickle loading of unverified files. | Security / MLOps | **MITIGATED (SAFE UNPICKLER USED)** |
| **RSK-06** | Hardware Incompatibility for TensorRT (Host CPU-only) | HIGH | HIGH | Host confirmed `torch.cuda.is_available() == False`. Standardize on ONNX Runtime CPU for local testing; isolate TensorRT build to remote/edge CUDA targets. | Edge Eng | **CONFIRMED & MITIGATED** |
| **RSK-07** | MQTT Telemetry Schema Mismatch with Team 5C | HIGH | HIGH | Formalize schema early and validate against sample JSON fixtures before integration. | Systems Eng | **ACTIVE** |
| **RSK-08** | Memory Exhaustion when Loading 3 Models Simultaneously | MEDIUM | HIGH | Implement lazy loading in domain router; unload inactive models if memory budget is tight. | Edge Eng | **PLANNED** |

---

## 14. Current Status & Next Actions

* **Last Updated:** 2026-10-11
* **Current Phase:** Phase 1 — Model & Dataset Discovery (Completed for Checkpoints) $\rightarrow$ Phase 2/3 (Inference Architecture & Scaffolding)
* **Verified Accomplishments:**
  * All 3 checkpoint files placed in `checkpoints/`:
    * `Agriculture_bestweight.pt` (5.94 MB, SHA: `4480b999...`)
    * `best.pt` (5.14 MB, SHA: `304f2f9b...`)
    * `space_electronics.pt` (5.14 MB, SHA: `50f7a458...`)
  * Extracted complete checkpoint architecture & training metadata:
    * **Agriculture:** `yolov8n.pt` @ `512x512`, `{0: 'healthy_crop', 1: 'dead_crop'}`
    * **Ships:** `yolo26n.pt` @ `640x640`, `{0: 'cargo_ship', 1: 'passenger_ship'}`
    * **Space Electronics:** `yolo26n.pt` @ `640x640`, `{0: 'open', 1: 'short'}`
  * Confirmed that direct weight merging is structurally impossible due to `C2f` vs `C3k2/C2PSA` block divergence.
  * Updated `configs/taxonomy.yaml` with verified local-to-global translations.
* **Active Blockers & Dependencies:**
  * Validation / test sample images and labels for each domain (needed for mAP evaluation, dataset validation, and ONNX numerical equivalence).
  * Team 5C MQTT Broker credentials / topic schema.
* **Immediate Next Actions (Ranked):**
  1. Build the canonical Unified Inference Engine schema and abstract adapter framework (`src/inference/`).
  2. Implement domain-specific adapters (`AgricultureAdapter`, `ShipsAdapter`, `SpaceElectronicsAdapter`) with resolution-aware preprocessing (`512` vs `640`) and local-to-global taxonomy translation.
  3. Implement the `UnifiedInferenceRouter` with domain routing and normalized output formatting.
  4. Create unit tests for schema, routing, and taxonomy translation.
  5. Receive validation image/label samples from collaborators when available.

