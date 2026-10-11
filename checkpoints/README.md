# Checkpoints Directory
## Team 5B – Edge Inference & Hardware Integration

Place original domain-specific trained weights (`.pt`) into this folder.

### Expected Files:
1. `Agriculture_bestweight.pt` (Agriculture domain model)
2. `best (1).pt` (Ships domain model)
3. `space_electronics.pt` (Space Electronics domain model)

### Checkpoint Handling Rules:
* **Never overwrite original weights:** All conversions (ONNX, TensorRT engines, quantized weights) must be written to `artifacts/exports/`.
* **Security Notice:** PyTorch `.pt` files use pickle serialization. Do not perform unchecked deserialization from unverified sources. Checkpoints will be inspected using safe inspect scripts prior to full model loading.
* **Large Files:** `.pt` files are excluded from Git tracking via `.gitignore`. Store backups and compute SHA-256 hashes for auditing in `docs/ARTIFACT_INVENTORY.md`.
