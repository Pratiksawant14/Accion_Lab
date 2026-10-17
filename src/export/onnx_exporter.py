"""Automated ONNX Model Exporter for Team 5B.

Conforms to Requirement REQ-05 and Milestone B4:
- Converts PyTorch (.pt) checkpoints to ONNX format.
- Configurable input resolution (512x512 for Agriculture, 640x640 for Ships & Space Electronics).
- Configurable dynamic batching and opset version.
- Validates structural ONNX graph integrity using onnx.checker.
- Preserves checkpoints/ source files and safely routes exports to artifacts/exports/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import onnx

logger = logging.getLogger("ONNXExporter")


@dataclass
class ExportResult:
    """Metadata result from an ONNX model export operation."""

    domain: str
    checkpoint_path: str
    onnx_path: str
    input_resolution: Tuple[int, int]
    opset: int
    dynamic_axes: bool
    file_size_bytes: int
    file_size_mb: float
    sha256_hash: str
    export_duration_sec: float
    is_valid_onnx: bool
    graph_inputs: List[str]
    graph_outputs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ONNXExporter:
    """Manages exporting PyTorch YOLO checkpoints to ONNX format."""

    def __init__(self, exports_dir: Union[str, Path] = "artifacts/exports"):
        self.exports_dir = Path(exports_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def _compute_sha256(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def export(
        self,
        checkpoint_path: Union[str, Path],
        domain: str,
        target_filename: Optional[str] = None,
        imgsz: int = 640,
        dynamic: bool = False,
        opset: int = 12,
        simplify: bool = False,
    ) -> ExportResult:
        """Exports a YOLO .pt checkpoint to ONNX."""
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")

        domain_key = domain.lower()
        output_filename = target_filename or f"{domain_key}.onnx"
        destination_onnx = self.exports_dir / output_filename

        logger.info("Exporting %s (%s) to ONNX at %s (imgsz=%d, opset=%d)...", domain_key, ckpt.name, destination_onnx, imgsz, opset)
        t_start = time.perf_counter()

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Ultralytics required for ONNX export. Run: pip install ultralytics") from e

        model = YOLO(str(ckpt.resolve()))
        raw_exported = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=dynamic,
            opset=opset,
            simplify=simplify,
        )

        exported_path = Path(raw_exported)
        if not exported_path.is_file():
            raise FileNotFoundError(f"Expected exported file at {exported_path}, but not found.")

        # Ensure model is safely moved to artifacts/exports without polluting checkpoints/
        if exported_path.resolve() != destination_onnx.resolve():
            shutil.copy2(exported_path, destination_onnx)
            # If the exported file was placed in checkpoints/, clean up the copy there
            if exported_path.parent.resolve() == ckpt.parent.resolve():
                try:
                    exported_path.unlink()
                except Exception:
                    pass

        export_duration = time.perf_counter() - t_start

        # Validate ONNX graph integrity
        onnx_model = onnx.load(str(destination_onnx))
        onnx.checker.check_model(onnx_model)
        is_valid = True

        graph_inputs = [inp.name for inp in onnx_model.graph.input]
        graph_outputs = [out.name for out in onnx_model.graph.output]

        file_size = destination_onnx.stat().st_size
        sha256 = self._compute_sha256(destination_onnx)

        logger.info(
            "ONNX export succeeded: %s (%.2f MB, SHA256: %s... in %.2fs)",
            destination_onnx.name,
            file_size / (1024 * 1024),
            sha256[:8],
            export_duration,
        )

        return ExportResult(
            domain=domain_key,
            checkpoint_path=str(ckpt.resolve()),
            onnx_path=str(destination_onnx.resolve()),
            input_resolution=(imgsz, imgsz),
            opset=opset,
            dynamic_axes=dynamic,
            file_size_bytes=file_size,
            file_size_mb=round(file_size / (1024 * 1024), 2),
            sha256_hash=sha256,
            export_duration_sec=round(export_duration, 2),
            is_valid_onnx=is_valid,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
        )
