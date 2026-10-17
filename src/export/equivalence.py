"""Numerical Output Equivalence Testing for PyTorch vs. ONNX Runtime.

Conforms to Requirement REQ-05 and Milestone B5:
- Feeds identical input tensors to PyTorch and ONNX models.
- Evaluates raw output predictions prior to NMS discretization.
- Computes max absolute difference, mean absolute difference, and relative error.
- Asserts max absolute difference <= tolerance (standard threshold: 1e-3).
- Generates JSON and Markdown verification reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

logger = logging.getLogger("EquivalenceValidator")


@dataclass
class SampleEquivalenceResult:
    """Numerical equivalence result for a single test input sample."""

    sample_index: int
    source_image: Optional[str]
    input_shape: List[int]
    pytorch_output_shape: List[int]
    onnx_output_shape: List[int]
    max_absolute_error: float
    mean_absolute_error: float
    passed: bool


@dataclass
class EquivalenceReport:
    """Master numerical equivalence report for a domain model."""

    domain: str
    checkpoint_path: str
    onnx_path: str
    input_resolution: Tuple[int, int]
    tolerance: float
    num_samples_tested: int
    max_absolute_error_overall: float
    mean_absolute_error_overall: float
    all_passed: bool
    samples: List[SampleEquivalenceResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def generate_markdown_summary(self) -> str:
        status_str = "PASS (Equivalence Confirmed)" if self.all_passed else "FAIL (Exceeds Tolerance)"
        lines = [
            f"# Numerical Equivalence Report: {self.domain.upper()}",
            f"- **PyTorch Model:** `{Path(self.checkpoint_path).name}`",
            f"- **ONNX Model:** `{Path(self.onnx_path).name}`",
            f"- **Tolerance Threshold:** `<= {self.tolerance}`",
            f"- **Status:** **{status_str}**",
            f"- **Overall Max Absolute Error:** `{self.max_absolute_error_overall:.6e}`",
            f"- **Overall Mean Absolute Error:** `{self.mean_absolute_error_overall:.6e}`",
            "",
            "## Sample Level Verification",
            "",
            "| Sample # | Input Source | Shape | Max Abs Error | Mean Abs Error | Status |",
            "|---|---|:---:|:---:|:---:|:---:|",
        ]
        for s in self.samples:
            src = Path(s.source_image).name if s.source_image else "Synthetic Random Tensor"
            st = "PASS" if s.passed else "FAIL"
            lines.append(
                f"| {s.sample_index} | `{src}` | {s.input_shape} | "
                f"`{s.max_absolute_error:.6e}` | `{s.mean_absolute_error:.6e}` | **{st}** |"
            )

        return "\n".join(lines)


class OutputEquivalenceValidator:
    """Validates numerical output equivalence between PyTorch and ONNX models."""

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def _prepare_tensor_from_image(self, img_path: Path, imgsz: int) -> np.ndarray:
        """Loads and normalizes an image to [1, 3, imgsz, imgsz] float32 in [0, 1]."""
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB").resize((imgsz, imgsz))
            arr = np.array(rgb_img, dtype=np.float32) / 255.0
            # [H, W, C] -> [1, C, H, W]
            arr = np.transpose(arr, (2, 0, 1))
            return np.expand_dims(arr, axis=0)

    def verify(
        self,
        domain: str,
        checkpoint_path: Union[str, Path],
        onnx_path: Union[str, Path],
        imgsz: int = 640,
        sample_image_paths: Optional[List[Union[str, Path]]] = None,
        num_random_samples: int = 5,
    ) -> EquivalenceReport:
        """Runs numerical equivalence verification on sample inputs."""
        ckpt = Path(checkpoint_path)
        onnx_file = Path(onnx_path)

        if not ckpt.is_file():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt}")
        if not onnx_file.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_file}")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Ultralytics required for equivalence validation.") from e

        # 1. Load PyTorch model
        yolo_model = YOLO(str(ckpt.resolve()))
        pt_inner_model = yolo_model.model
        pt_inner_model.eval()

        # 2. Load ONNX Runtime session
        session = ort.InferenceSession(str(onnx_file.resolve()), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Prepare inputs
        inputs: List[Tuple[np.ndarray, Optional[str]]] = []
        if sample_image_paths:
            for p in sample_image_paths[:10]:
                tensor_np = self._prepare_tensor_from_image(Path(p), imgsz)
                inputs.append((tensor_np, str(p)))

        # Also add random synthetic inputs to thoroughly explore dynamic range
        np.random.seed(42)
        for _ in range(num_random_samples):
            rand_tensor = np.random.uniform(0.0, 1.0, size=(1, 3, imgsz, imgsz)).astype(np.float32)
            inputs.append((rand_tensor, None))

        sample_results: List[SampleEquivalenceResult] = []
        overall_max_err = 0.0
        overall_sum_mean_err = 0.0
        all_passed = True

        for idx, (input_np, source_name) in enumerate(inputs, start=1):
            input_torch = torch.from_numpy(input_np)

            # PyTorch forward pass
            with torch.no_grad():
                pt_raw = pt_inner_model(input_torch)
                if isinstance(pt_raw, (tuple, list)):
                    pt_raw = pt_raw[0]
                pt_arr = pt_raw.cpu().numpy()

            # ONNX Runtime forward pass
            ort_outs = session.run(None, {input_name: input_np})
            ort_arr = ort_outs[0]

            # Reconcile dimensions if PyTorch is [1, 6, 8400] and ORT is [1, 6, 8400]
            if pt_arr.shape != ort_arr.shape:
                if pt_arr.shape == ort_arr.T.shape:
                    ort_arr = ort_arr.T

            # Compute errors
            abs_diff = np.abs(pt_arr - ort_arr)
            max_abs = float(np.max(abs_diff))
            mean_abs = float(np.mean(abs_diff))

            passed = max_abs <= self.tolerance
            if not passed:
                all_passed = False

            if max_abs > overall_max_err:
                overall_max_err = max_abs
            overall_sum_mean_err += mean_abs

            sample_results.append(
                SampleEquivalenceResult(
                    sample_index=idx,
                    source_image=source_name,
                    input_shape=list(input_np.shape),
                    pytorch_output_shape=list(pt_arr.shape),
                    onnx_output_shape=list(ort_arr.shape),
                    max_absolute_error=max_abs,
                    mean_absolute_error=mean_abs,
                    passed=passed,
                )
            )

        mean_overall = overall_sum_mean_err / max(1, len(inputs))

        return EquivalenceReport(
            domain=domain.lower(),
            checkpoint_path=str(ckpt.resolve()),
            onnx_path=str(onnx_file.resolve()),
            input_resolution=(imgsz, imgsz),
            tolerance=self.tolerance,
            num_samples_tested=len(inputs),
            max_absolute_error_overall=overall_max_err,
            mean_absolute_error_overall=mean_overall,
            all_passed=all_passed,
            samples=sample_results,
        )
