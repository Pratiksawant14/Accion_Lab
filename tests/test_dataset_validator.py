"""Unit tests for automated dataset validator (Milestone B1 / REQ-02)."""

import json
from pathlib import Path
from PIL import Image
import pytest

from src.data.validator import DatasetValidator, ValidationReport


@pytest.fixture
def temp_dataset(tmp_path: Path) -> Path:
    """Creates a temporary miniature dataset structure with valid & corrupted samples."""
    dataset_dir = tmp_path / "mock_dataset"
    img_dir = dataset_dir / "images" / "val"
    lbl_dir = dataset_dir / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # 1. Valid image and valid localized label
    valid_img_path = img_dir / "sample_valid.jpg"
    img = Image.new("RGB", (100, 100), color="green")
    img.save(valid_img_path)

    valid_lbl_path = lbl_dir / "sample_valid.txt"
    # class 0, centered, small box
    valid_lbl_path.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    # 2. Pseudo-box image and label
    pseudo_img_path = img_dir / "sample_pseudo.jpg"
    img_p = Image.new("RGB", (100, 100), color="blue")
    img_p.save(pseudo_img_path)

    pseudo_lbl_path = lbl_dir / "sample_pseudo.txt"
    # class 1, full frame box (w=0.98, h=0.99)
    pseudo_lbl_path.write_text("1 0.5 0.5 0.98 0.99\n", encoding="utf-8")

    # 3. Corrupt image (0 bytes)
    corrupt_img_path = img_dir / "sample_corrupt.jpg"
    corrupt_img_path.write_bytes(b"")

    corrupt_lbl_path = lbl_dir / "sample_corrupt.txt"
    corrupt_lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # 4. Invalid coordinates in label
    inv_img_path = img_dir / "sample_invalid_coords.jpg"
    img_i = Image.new("RGB", (100, 100), color="red")
    img_i.save(inv_img_path)

    inv_lbl_path = lbl_dir / "sample_invalid_coords.txt"
    # x=1.5 (out of range > 1.0)
    inv_lbl_path.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")

    # 5. Out of range class
    out_img_path = img_dir / "sample_out_class.jpg"
    img_o = Image.new("RGB", (100, 100), color="yellow")
    img_o.save(out_img_path)

    out_lbl_path = lbl_dir / "sample_out_class.txt"
    # class 99 is invalid for agriculture (only 0 and 1 allowed)
    out_lbl_path.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # 6. Unpaired image (no label)
    unpaired_img = img_dir / "sample_unpaired.jpg"
    img_u = Image.new("RGB", (100, 100), color="white")
    img_u.save(unpaired_img)

    # 7. Orphan label (no image)
    orphan_lbl = lbl_dir / "sample_orphan.txt"
    orphan_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    return dataset_dir


def test_validator_detects_all_anomalies(temp_dataset: Path):
    validator = DatasetValidator(domain="agriculture")
    report = validator.validate_dataset(temp_dataset, splits=["val"])

    assert not report.is_valid_for_training
    val_split = report.splits["val"]

    # Image validation
    assert val_split.total_images == 6
    assert val_split.valid_images == 5
    assert val_split.corrupt_images == 1

    # Labels and pairings
    assert val_split.total_labels == 6
    assert val_split.images_missing_labels == 1
    assert val_split.orphan_labels == 1

    # Bounding box sanity
    assert val_split.pseudo_boxes == 1
    assert val_split.invalid_coordinate_boxes == 1
    assert val_split.out_of_range_classes == 1

    # Markdown generation
    md = report.generate_markdown_summary()
    assert "# Dataset Pre-Flight Validation Report: AGRICULTURE" in md
    assert "FAIL" in md


def test_validator_clean_dataset(tmp_path: Path):
    clean_dir = tmp_path / "clean_dataset"
    img_dir = clean_dir / "images" / "train"
    lbl_dir = clean_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = Image.new("RGB", (512, 512), color="green")
    img.save(img_dir / "img1.png")
    (lbl_dir / "img1.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    validator = DatasetValidator(domain="ships")
    report = validator.validate_dataset(clean_dir, splits=["train"])

    assert report.is_valid_for_training
    train_split = report.splits["train"]
    assert train_split.is_clean
    assert train_split.corrupt_images == 0
    assert train_split.corrupt_labels == 0
    assert train_split.invalid_coordinate_boxes == 0
    assert train_split.out_of_range_classes == 0
    assert train_split.pseudo_boxes == 0
