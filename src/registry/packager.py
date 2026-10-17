"""Auditable Model Packaging & Release Manifest Generator (REQ-12 / Milestone B10).

Creates complete, tamper-evident release bundles containing PyTorch & ONNX models,
cryptographic SHA-256 hashes, dataset lineage, Git commit provenance,
training/evaluation metrics, and edge hardware benchmark metrics.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional
import zipfile
from pydantic import BaseModel, Field


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 cryptographic checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> Optional[str]:
    """Retrieve current Git commit hash if in a Git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=3
        ).strip()
        return commit
    except Exception:
        return None


class ArtifactChecksum(BaseModel):
    """File metadata with SHA-256 integrity hash."""

    filename: str
    sha256: str
    size_bytes: int


class ReleaseManifest(BaseModel):
    """Auditable release manifest for edge deployments."""

    release_id: str
    domain: str
    architecture: str
    version: str = "1.0.0"
    input_resolution: List[int]
    git_commit_hash: Optional[str] = None
    pt_checkpoint: ArtifactChecksum
    onnx_model: ArtifactChecksum
    dataset_provenance: Dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict)
    benchmark_telemetry: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_markdown(self) -> str:
        """Generate human-readable audit summary in Markdown."""
        lines = [
            f"# Release Manifest: {self.release_id}",
            f"- **Domain:** `{self.domain}`",
            f"- **Architecture:** `{self.architecture}`",
            f"- **Release Version:** `{self.version}`",
            f"- **Git Commit:** `{self.git_commit_hash or 'N/A'}`",
            f"- **Input Resolution:** `{self.input_resolution[0]}x{self.input_resolution[1]}`",
            f"- **Generated At:** `{self.timestamp}`",
            "",
            "## Model Artifact Integrity Checksums",
            "",
            "| Artifact | Filename | Size (Bytes) | SHA-256 Checksum |",
            "|---|---|:---:|---|",
            f"| PyTorch Checkpoint | `{self.pt_checkpoint.filename}` | {self.pt_checkpoint.size_bytes:,} | `{self.pt_checkpoint.sha256}` |",
            f"| ONNX Export | `{self.onnx_model.filename}` | {self.onnx_model.size_bytes:,} | `{self.onnx_model.sha256}` |",
            "",
            "## Evaluation Metrics",
            "",
        ]
        if self.evaluation_metrics:
            for k, v in sorted(self.evaluation_metrics.items()):
                lines.append(f"- **{k}:** `{v}`")
        else:
            lines.append("- *No evaluation metrics recorded.*")

        lines.extend(["", "## Edge Hardware Benchmark Metrics", ""])
        if self.benchmark_telemetry:
            for k, v in sorted(self.benchmark_telemetry.items()):
                lines.append(f"- **{k}:** `{v}`")
        else:
            lines.append("- *No benchmark metrics recorded.*")

        lines.extend(["", "## Dataset Lineage & Provenance", ""])
        if self.dataset_provenance:
            for k, v in sorted(self.dataset_provenance.items()):
                lines.append(f"- **{k}:** `{v}`")
        else:
            lines.append("- *No dataset provenance recorded.*")

        return "\n".join(lines)


class ModelPackager:
    """Packages model checkpoints, ONNX exports, and audit manifests into release bundles."""

    def __init__(self, output_root: str | Path = "artifacts/releases"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def package(
        self,
        domain: str,
        pt_path: str | Path,
        onnx_path: str | Path,
        architecture: str,
        input_resolution: List[int],
        version: str = "1.0.0",
        dataset_provenance: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[Dict[str, float]] = None,
        benchmark_telemetry: Optional[Dict[str, Any]] = None,
        create_zip: bool = True,
    ) -> ReleaseManifest:
        """Create a complete release package for a domain model."""
        pt_file = Path(pt_path)
        onnx_file = Path(onnx_path)

        if not pt_file.exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {pt_file}")
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")

        release_id = f"release_{domain}_v{version}"
        release_dir = self.output_root / release_id
        release_dir.mkdir(parents=True, exist_ok=True)

        # 1. Compute checksums
        pt_sha = compute_sha256(pt_file)
        onnx_sha = compute_sha256(onnx_file)

        pt_meta = ArtifactChecksum(
            filename=pt_file.name,
            sha256=pt_sha,
            size_bytes=pt_file.stat().st_size,
        )
        onnx_meta = ArtifactChecksum(
            filename=onnx_file.name,
            sha256=onnx_sha,
            size_bytes=onnx_file.stat().st_size,
        )

        # 2. Copy artifacts into release directory
        dest_pt = release_dir / pt_file.name
        dest_onnx = release_dir / onnx_file.name
        shutil.copy2(pt_file, dest_pt)
        shutil.copy2(onnx_file, dest_onnx)

        # 3. Create manifest
        manifest = ReleaseManifest(
            release_id=release_id,
            domain=domain,
            architecture=architecture,
            version=version,
            input_resolution=input_resolution,
            git_commit_hash=get_git_commit(),
            pt_checkpoint=pt_meta,
            onnx_model=onnx_meta,
            dataset_provenance=dataset_provenance or {},
            evaluation_metrics=evaluation_metrics or {},
            benchmark_telemetry=benchmark_telemetry or {},
        )

        # 4. Save manifest JSON & Markdown README
        manifest_json_path = release_dir / "manifest.json"
        manifest_json_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        readme_path = release_dir / "README.md"
        readme_path.write_text(manifest.to_markdown(), encoding="utf-8")

        # 5. Optionally create a compressed zip release archive
        if create_zip:
            zip_target = self.output_root / f"{release_id}.zip"
            with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in release_dir.iterdir():
                    zf.write(f, arcname=f.name)

        return manifest
