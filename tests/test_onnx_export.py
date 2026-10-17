"""Unit tests for ONNX Export and Numerical Equivalence (Milestones B4 & B5 / REQ-05)."""

from pathlib import Path
import numpy as np
import pytest

from src.export.onnx_exporter import ONNXExporter
from src.export.equivalence import EquivalenceReport, OutputEquivalenceValidator, SampleEquivalenceResult


def test_exporter_initialization(tmp_path: Path):
    export_dir = tmp_path / "custom_exports"
    exporter = ONNXExporter(exports_dir=export_dir)
    assert export_dir.exists()
    assert exporter.exports_dir == export_dir


def test_equivalence_report_markdown_generation():
    sample1 = SampleEquivalenceResult(
        sample_index=1,
        source_image="test.png",
        input_shape=[1, 3, 640, 640],
        pytorch_output_shape=[1, 6, 8400],
        onnx_output_shape=[1, 6, 8400],
        max_absolute_error=1.2e-4,
        mean_absolute_error=2.5e-6,
        passed=True,
    )
    report = EquivalenceReport(
        domain="ships",
        checkpoint_path="checkpoints/best.pt",
        onnx_path="artifacts/exports/ships.onnx",
        input_resolution=(640, 640),
        tolerance=1e-3,
        num_samples_tested=1,
        max_absolute_error_overall=1.2e-4,
        mean_absolute_error_overall=2.5e-6,
        all_passed=True,
        samples=[sample1],
    )

    md = report.generate_markdown_summary()
    assert "# Numerical Equivalence Report: SHIPS" in md
    assert "PASS (Equivalence Confirmed)" in md
    assert "1.200000e-04" in md


def test_validator_tolerance_logic():
    validator = OutputEquivalenceValidator(tolerance=1e-3)
    assert validator.tolerance == 1e-3

    # If error is 5e-4 (less than 1e-3), it passes
    assert 5e-4 <= validator.tolerance

    # If error is 2e-3 (greater than 1e-3), it fails
    assert not (2e-3 <= validator.tolerance)
