"""Unit tests for ModelPackager and ReleaseManifest (Milestone B10)."""

import json
from pathlib import Path
import pytest
import zipfile

from src.registry.packager import ModelPackager, ReleaseManifest, compute_sha256


class TestModelPackager:
    """Test suite for auditable model packaging."""

    def test_compute_sha256(self, tmp_path):
        dummy_file = tmp_path / "sample.bin"
        dummy_file.write_bytes(b"HELLO_WORLD_CHECKSUM_TEST")
        sha = compute_sha256(dummy_file)
        assert len(sha) == 64
        assert isinstance(sha, str)

    def test_missing_files_raise(self, tmp_path):
        packager = ModelPackager(output_root=tmp_path / "releases")
        missing_pt = tmp_path / "missing.pt"
        missing_onnx = tmp_path / "missing.onnx"

        with pytest.raises(FileNotFoundError):
            packager.package(
                domain="agriculture",
                pt_path=missing_pt,
                onnx_path=missing_onnx,
                architecture="YOLOv8n",
                input_resolution=[512, 512],
            )

    def test_package_creation(self, tmp_path):
        pt_file = tmp_path / "Agriculture_bestweight.pt"
        onnx_file = tmp_path / "agriculture.onnx"
        pt_file.write_bytes(b"MOCK_PT_WEIGHTS_123456789")
        onnx_file.write_bytes(b"MOCK_ONNX_MODEL_987654321")

        release_root = tmp_path / "releases"
        packager = ModelPackager(output_root=release_root)

        manifest = packager.package(
            domain="agriculture",
            pt_path=pt_file,
            onnx_path=onnx_file,
            architecture="YOLOv8n",
            input_resolution=[512, 512],
            version="1.0.0",
            dataset_provenance={"dataset_root": "Synthline/agriculture robotics", "total_images": 972},
            evaluation_metrics={"map50": 0.92, "map50_95": 0.76},
            benchmark_telemetry={"fps": 5.39, "peak_ram_mb": 399.4},
            create_zip=True,
        )

        assert isinstance(manifest, ReleaseManifest)
        assert manifest.release_id == "release_agriculture_v1.0.0"
        assert manifest.domain == "agriculture"
        assert len(manifest.pt_checkpoint.sha256) == 64
        assert len(manifest.onnx_model.sha256) == 64
        assert manifest.evaluation_metrics["map50"] == 0.92

        # Check files on disk
        release_dir = release_root / "release_agriculture_v1.0.0"
        assert (release_dir / "Agriculture_bestweight.pt").exists()
        assert (release_dir / "agriculture.onnx").exists()
        assert (release_dir / "manifest.json").exists()
        assert (release_dir / "README.md").exists()

        # Check zip archive
        zip_file = release_root / "release_agriculture_v1.0.0.zip"
        assert zip_file.exists()
        with zipfile.ZipFile(zip_file, "r") as zf:
            namelist = zf.namelist()
            assert "Agriculture_bestweight.pt" in namelist
            assert "agriculture.onnx" in namelist
            assert "manifest.json" in namelist
            assert "README.md" in namelist

        # Check JSON manifest content
        with open(release_dir / "manifest.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["domain"] == "agriculture"
        assert data["architecture"] == "YOLOv8n"

        # Check Markdown content
        md_text = (release_dir / "README.md").read_text(encoding="utf-8")
        assert "Release Manifest: release_agriculture_v1.0.0" in md_text
        assert "Model Artifact Integrity Checksums" in md_text
