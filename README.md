# Industry Project – Action Lab: Team 5B
## Edge Inference & Hardware Integration

Welcome to the Team 5B codebase for the Industry Project – Action Lab.

Our primary mission is:
1. **Unified Inference Orchestrator:** Integrating three domain-specific trained models (**Agriculture**, **Ships**, and **Space Electronics**) behind a unified inference routing and output normalization system.
2. **Automated Training & Evaluation Pipeline:** Building an end-to-end reproducible training workflow with automated dataset validation, checkpointing, evaluation, and packaging.
3. **Edge Optimization & Hardware Telemetry:** Exporting models to ONNX/TensorRT, benchmarking latency and throughput (p50/p95/p99 FPS), simulating edge constraints, and streaming telemetry to Team 5C via MQTT.

---

## Authoritative Documentation

All architectural decisions, requirements traceability, checkpoint health, class taxonomy, risks, and next steps are centrally maintained in:
👉 **[`PROJECT_SOURCE_OF_TRUTH.md`](PROJECT_SOURCE_OF_TRUTH.md)**

Consult this document before making any architectural or implementation changes.

---

## Project Structure

```text
d:/Accion_Lab/
├── PROJECT_SOURCE_OF_TRUTH.md   # Authoritative, living source of truth
├── README.md                    # Project overview and navigation
├── requirements.txt             # Core Python dependencies
├── .gitignore                   # Excludes checkpoints, datasets, cache
├── configs/                     # System & domain configurations
│   ├── taxonomy.yaml            # Global class taxonomy & domain mapping
│   ├── datasets/                # Domain dataset configurations (data.yaml)
│   └── training/                # Training run hyperparameter configs
├── checkpoints/                 # Storage for original .pt weight files
│   └── README.md                # Placement & security guidance
├── notebooks/                   # Jupyter notebooks (exploration & training)
├── src/                         # Modular application source code
│   ├── data_validation/         # Dataset integrity & label validation
│   ├── training/                # Automated training runners
│   ├── evaluation/              # mAP metrics & validation harness
│   ├── inference/               # Unified inference pipeline & router
│   │   ├── adapters/            # Domain-specific model adapters
│   │   ├── router.py            # Domain routing engine
│   │   ├── schema.py            # Canonical inference schema
│   │   └── taxonomy.py          # Class translation utilities
│   ├── export/                  # ONNX / TensorRT export utilities
│   ├── benchmarking/            # Latency, FPS, percentiles harness
│   ├── edge_simulation/         # Resource constraint simulation
│   ├── telemetry/               # MQTT publisher, buffer, reconnect
│   └── registry/                # Packaging, manifests, versioning
├── tests/                       # Automated test suite (pytest)
├── scripts/                     # Operational utilities & inspect scripts
├── artifacts/                   # Output storage (exports, manifests)
│   └── exports/                 # ONNX / TensorRT engines
├── reports/                     # Benchmarks, equivalence, metrics
└── docs/                        # Specifications and inventory
    └── ARTIFACT_INVENTORY.md    # Real-time tracking of received files
```

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Checkpoint Placement
Copy your domain-specific PyTorch weights into `checkpoints/`:
* Agriculture: `checkpoints/Agriculture_bestweight.pt`
* Ships: `checkpoints/best (1).pt`
* Space Electronics: `checkpoints/space_electronics.pt`

Verify placement and inspect headers safely:
```bash
python scripts/inspect_checkpoints.py
```
