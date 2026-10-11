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
| **REQ-01** | Model training using configurable hyperparameters | Project Brief §1.C, Roadmap B2 | **NOT STARTED** | Automated test running train script with test YAML config | None | Dataset, GPU/CPU environment, YOLO framework | Confirm exact base YOLO version (v8/v9/26) |
| **REQ-02** | Dataset validation before training (labels, image integrity, bounding box sanity) | Project Brief §1.C, Roadmap B1 | **NOT STARTED** | Unit tests with synthetic corrupted annotations & images | None | Python schema validator | Verification rules for full-image pseudo-boxes |
| **REQ-03** | Training checkpointing and resume support | Project Brief §1.C, Roadmap B2 | **NOT STARTED** | Interrupt training run and execute `--resume` flag validation | None | Checkpoint I/O handler | Storage path conventions |
| **REQ-04** | Support for YOLO Nano (n), Small (s), and Medium (m) architectures | Project Brief §1.C | **NOT STARTED** | Config-driven instantiation and parameter verification | None | Model definition files | Confirm if architecture choice is per-domain |
| **REQ-05** | ONNX export and output-equivalence validation | Project Brief §1.C, Roadmap B4, B5 | **NOT STARTED** | Numerical equivalence test (`numpy.allclose` or max abs tolerance) between PyTorch & ONNX | None | `onnx`, `onnxruntime` | Agreed numerical tolerance threshold (e.g. 1e-3) |
| **REQ-06** | TensorRT FP16 and INT8 export where supported | Project Brief §1.C, Roadmap B7 | **NOT STARTED** | Engine compilation check on compatible NVIDIA target | None | CUDA, TensorRT runtime, TensorRT engine builder | Host vs. target edge device hardware compatibility |
| **REQ-07** | INT8 calibration for quantized inference | Project Brief §1.C, Roadmap B7 | **NOT STARTED** | Calibration dataset execution and mAP degradation check | None | Representative calibration dataset | Minimum number of calibration frames required |
| **REQ-08** | Edge-resource constraint simulation (CPU throttle, memory caps, network delay) | Project Brief §1.C, Roadmap B8 | **NOT STARTED** | Benchmark under throttled cgroups/process priority | None | Resource simulation utility/cgroups | Hardware profile specifications of target edge unit |
| **REQ-09** | MQTT telemetry publication using agreed JSON schema | Project Brief §1.C, Roadmap B9 | **NOT STARTED** | Schema validation against paho-mqtt test subscriber | None | Broker credentials, schema agreement with Team 5C | Broker address, port, and QoS level |
| **REQ-10** | Telemetry buffering (1,000 capacity), exp backoff reconnection, 60s failure warning | Project Brief §1.C, Roadmap B9 | **NOT STARTED** | Chaos test: disconnect broker, publish 1,200 events, verify backoff & buffer flush | None | Memory buffer implementation | Overflow policy: FIFO drop oldest vs. drop newest? |
| **REQ-11** | Performance benchmarking: >=100 frames, FPS, p50/p95/p99 latency, mAP@0.5, mAP@0.5:0.95 | Project Brief §1.C, Roadmap B6 | **NOT STARTED** | Automated benchmark harness on representative test set | None | Benchmark dataset, evaluation harness | Does 50ms latency target include MQTT transmission? |
| **REQ-12** | Model versioning, registration, packaging, and auditability | Project Brief §1.C, Roadmap B10 | **NOT STARTED** | Manifest verification with checksums, Git commit SHA, and hyperparams | None | Registry schema | Local filesystem vs. remote artifact store |
| **REQ-13** | Integrated demonstration with Team 5A (inputs) and Team 5C (telemetry consumer) | Project Brief §1.C, Roadmap B11 | **NOT STARTED** | End-to-end integration test with live stream and telemetry consumer | None | Cross-team interface compliance | Integration schedule & demonstration environment |

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

### 4.2 Local vs. Global Class ID Reconciliation
*Warning:* Domain-specific checkpoints typically assign local classes beginning at index 0. For example:
* Agriculture model: Local 0 = `healthy_crop`, Local 1 = `dead_crop`.
* Ships model: Local 0 = `cargo_ship`, Local 1 = `passenger_ship` (or vice-versa).
* Space Electronics model: Local 0 / 1 mapping between `open` and `short` must be extracted directly from `data.yaml`.

**Rule:** Model adapters **must not** emit raw local class IDs. The Unified Inference Router will translate local class IDs into Global IDs using verified domain translation tables:

```python
# Conceptual translation mapping
CLASS_MAPPING_TABLE = {
    "agriculture": {0: 0, 1: 1},        # Local -> Global
    "ships": {0: 2, 1: 3},              # Subject to confirmation via data.yaml
    "space_electronics": {0: 4, 1: 5}   # Subject to confirmation via data.yaml
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
| **Agriculture** | 4,000 synthetic (3,608 PASS, 392 REVIEW); 2,525 train / 721 val / 362 test | **NO (Missing)** | 0 | 0 | Pending acquisition from model owner / Team 5A |
| **Ships** | 4,000 PASS; 2,800 train / 800 val / 400 test | **NO (Missing)** | 0 | 0 | Pending acquisition from model owner / Team 5A |
| **Space Electronics** | Pipeline generation completion unconfirmed | **NO (Missing)** | 0 | 0 | Pending confirmation of generation status |

### Known Label Limitations & Risks:
* **Full-Image Pseudo-Boxes:** Historical notes suggest some synthetic datasets utilized pseudo-bounding boxes spanning the entire image rather than tight localized objects. If present, mAP localization metrics will be artificially distorted. Validation checks must flag bounding boxes where `width > 0.95` and `height > 0.95`.

---

## 7. Model Checkpoint Inventory & Health

| Model Assignment | Filename | Current Workspace Path | File Size | Architecture / Version | Classes Count | Checkpoint Provenance | Verification Status |
|---|---|---|---|---|---|---|---|
| **Agriculture** | `Agriculture_bestweight.pt` | `checkpoints/Agriculture_bestweight.pt` | Unknown | Unverified (YOLOv8/v9/26?) | Expected 2 | Model Owner (Agriculture) | **MISSING LOCALLY** |
| **Ships** | `best (1).pt` | `checkpoints/best (1).pt` | Unknown | Unverified (YOLOv8/v9/26?) | Expected 2 | Model Owner (Ships) | **MISSING LOCALLY** |
| **Space Electronics** | `space_electronics.pt` | `checkpoints/space_electronics.pt` | Unknown | Unverified (YOLOv8/v9/26?) | Expected 2 | Model Owner (Electronics) | **MISSING LOCALLY** |

### Checkpoint Security & Integrity Rules:
* PyTorch `.pt` files use pickle serialization which can execute arbitrary code upon deserialization.
* **Rule:** Do not invoke blind `torch.load` with arbitrary weights from unverified sources without inspecting checkpoint structure using `torch.serialization.safe_globals` or safe extraction utilities.
* Original checkpoints must be archived as read-only. Converted models (ONNX, TensorRT) must be stored in `artifacts/exports/` and never overwrite `.pt` source files.

---

## 8. Artifacts & Information Request Tracker

| Requested Artifact | Source | Criticality | Status | Received Date | File Path / Hash |
|---|---|---|---|---|---|
| `Agriculture_bestweight.pt` | Agriculture Model Owner | CRITICAL | **AWAITING** | - | - |
| `best (1).pt` (Ships) | Ships Model Owner | CRITICAL | **AWAITING** | - | - |
| `space_electronics.pt` | Space Electronics Model Owner | CRITICAL | **AWAITING** | - | - |
| Agriculture `data.yaml` & Notebook | Agriculture Model Owner | HIGH | **AWAITING** | - | - |
| Ships `data.yaml` & Notebook | Ships Model Owner | HIGH | **AWAITING** | - | - |
| Electronics `data.yaml` & Notebook | Space Electronics Model Owner | HIGH | **AWAITING** | - | - |
| Original Written Requirements Doc | Project Lead | HIGH | **AWAITING** | - | - |
| Team 5C MQTT Schema Spec | Team 5C Lead | MEDIUM | **AWAITING** | - | - |
| Validation Sample Images & Labels | Team 5A | HIGH | **AWAITING** | - | - |

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
| **B0** | Environment & Dataset Setup | Virtual environment, dependencies, folder structure, checkpoint placement | **IN PROGRESS** | Git repo initialized; Python 3.13.7 & PyTorch 2.10 (CPU) verified; awaiting checkpoints |
| **B1** | Dataset Validation | Automated validator for image/label integrity & class counts | **NOT STARTED** | Test suite passes with valid & invalid dataset samples |
| **B2** | Training Pipeline | Automated training runner with checkpointing & resume support | **NOT STARTED** | Reproducible training execution on Ships domain |
| **B3** | Model Evaluation | Automated evaluation harness computing mAP@0.5 & mAP@0.5:0.95 | **NOT STARTED** | Metric reports generated matching manual baselines |
| **B4** | ONNX Export | Automated export pipeline for trained PyTorch checkpoints | **NOT STARTED** | Valid `.onnx` models generated with correct input/output heads |
| **B5** | Output Equivalence | PyTorch vs. ONNX numerical equivalence testing harness | **NOT STARTED** | Max absolute difference $\le 10^{-3}$ on identical test inputs |
| **B6** | Benchmarking | Automated latency & FPS harness (p50/p95/p99 across $\ge 100$ frames) | **NOT STARTED** | Benchmark report with reproducible percentile measurements |
| **B7** | TensorRT Support | FP16/INT8 TensorRT engine generation & calibration | **BLOCKED ON HOST** | Host is CPU-only (`torch.cuda.is_available() == False`); requires CUDA target |
| **B8** | Edge Simulation | Resource-throttled edge execution environment | **NOT STARTED** | Inference performance validated under simulated constraints |
| **B9** | MQTT Integration | Telemetry publisher with buffering (1,000 msgs) & reconnection | **NOT STARTED** | Broker disconnect/reconnect test passes with zero message loss |
| **B10** | Registry & Packaging | Artifact packaging with audit manifest, checksums, and metadata | **NOT STARTED** | Reproducible package load and verification |
| **B11** | Integrated Demo | Live end-to-end multi-domain edge inference demonstration | **NOT STARTED** | Live demonstration verified with Team 5A & Team 5C |

---

## 13. Risk Register & Mitigation Strategy

| Risk ID | Description | Likelihood | Impact | Mitigation Strategy | Owner | Status |
|:---:|---|:---:|:---:|---|:---:|:---:|
| **RSK-01** | Missing Checkpoint Files in workspace | HIGH | HIGH | Establish clear folder structure and request checkpoints from user immediately. | Team 5B Lead | **ACTIVE BLOCKER** |
| **RSK-02** | Unverified Local Class ID Order (e.g. `open` vs `short`) | HIGH | HIGH | Require `data.yaml` from model owners before enabling global class mapping. | Team 5B Lead | **ACTIVE** |
| **RSK-03** | YOLO Version Discrepancy (YOLO26 vs v8/v9) | MEDIUM | MEDIUM | Abstract model loading behind `BaseModelAdapter`; clarify version with Project Lead. | MLOps Eng | **ACTIVE** |
| **RSK-04** | Synthetic Data Pseudo-Bounding Boxes | MEDIUM | MEDIUM | Implement automated bounding box geometry validation to detect full-image boxes. | Data Eng | **MONITORED** |
| **RSK-05** | Checkpoint Deserialization Vulnerabilities | LOW | HIGH | Inspect checkpoint structure safely; avoid unsafe pickle loading of unverified files. | Security / MLOps | **ACTIVE** |
| **RSK-06** | Hardware Incompatibility for TensorRT (Host CPU-only) | HIGH | HIGH | Host confirmed `torch.cuda.is_available() == False`. Standardize on ONNX Runtime CPU for local testing; isolate TensorRT build to remote/edge CUDA targets. | Edge Eng | **CONFIRMED & MITIGATED** |
| **RSK-07** | MQTT Telemetry Schema Mismatch with Team 5C | HIGH | HIGH | Formalize schema early and validate against sample JSON fixtures before integration. | Systems Eng | **ACTIVE** |
| **RSK-08** | Memory Exhaustion when Loading 3 Models Simultaneously | MEDIUM | HIGH | Implement lazy loading in domain router; unload inactive models if memory budget is tight. | Edge Eng | **PLANNED** |

---

## 14. Current Status & Next Actions

* **Last Updated:** 2026-10-11
* **Current Phase:** Phase 0 — Onboarding, Environment Discovery & Source-of-Truth Documentation
* **Verified Accomplishments:**
  * Workspace `d:/Accion_Lab` inspected: clean directory confirmed.
  * Git repository initialized in `d:/Accion_Lab`.
  * Host environment verified: Python 3.13.7, PyTorch 2.10.0 (CPU mode, CUDA unavailable).
  * Created canonical `PROJECT_SOURCE_OF_TRUTH.md` establishing project identity, requirements traceability, architecture strategy, class taxonomy, and risk register.
  * Created `docs/ARTIFACT_INVENTORY.md` to track incoming checkpoints, notebooks, and dataset descriptors.
  * Created `configs/taxonomy.yaml` with global taxonomy (classes 0–5) and domain mappings.
  * Created `checkpoints/README.md` and safe checkpoint inspector `scripts/inspect_checkpoints.py`.
  * Created `requirements.txt` and `.gitignore`.
* **Active Work:**
  * Ready to receive the 3 checkpoint `.pt` files and domain configurations.
* **Active Blockers:**
  * Checkpoint files (`Agriculture_bestweight.pt`, `best (1).pt`, `space_electronics.pt`) are not yet placed in `checkpoints/`.
  * Training notebooks (`.ipynb`) and `data.yaml` descriptors have not yet been provided.
* **Immediate Next Actions (Ranked):**
  1. User copies the 3 checkpoint files into `checkpoints/`.
  2. Execute `python scripts/inspect_checkpoints.py` to verify hashes, sizes, and internal metadata.
  3. Obtain and review `data.yaml` and training notebooks for the three domains to confirm local-to-global class mappings and exact model architectures.
  4. Build baseline domain model adapters in `src/inference/adapters/`.

