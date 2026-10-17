"""Automated Dataset Pre-Flight & Sanity Validator for Team 5B.

Conforms to Requirement REQ-02 and Milestone B1:
- Validates image integrity and readability (corrupted or unreadable image files).
- Validates YOLO label format and coordinate boundaries (0.0 <= x, y, w, h <= 1.0).
- Verifies class index consistency against domain taxonomy.
- Automatically flags full-frame synthetic pseudo-bounding boxes (w > 0.95 and h > 0.95).
- Detects unpaired images and orphan label files.
- Generates structured JSON and Markdown audit reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from PIL import Image

from ..inference.schema import Domain
from ..inference.taxonomy import taxonomy_manager

logger = logging.getLogger("DatasetValidator")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitValidationResult:
    """Validation metrics for a single dataset split (e.g. train, val, test)."""

    split_name: str
    total_images: int = 0
    valid_images: int = 0
    corrupt_images: int = 0
    corrupt_image_files: List[str] = field(default_factory=list)

    total_labels: int = 0
    valid_labels: int = 0
    corrupt_labels: int = 0
    corrupt_label_files: List[str] = field(default_factory=list)

    images_missing_labels: int = 0
    orphan_labels: int = 0

    total_bounding_boxes: int = 0
    pseudo_boxes: int = 0
    invalid_coordinate_boxes: int = 0
    out_of_range_classes: int = 0

    class_distribution: Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def pseudo_box_percentage(self) -> float:
        if self.total_bounding_boxes == 0:
            return 0.0
        return round((self.pseudo_boxes / self.total_bounding_boxes) * 100.0, 2)

    @property
    def is_clean(self) -> bool:
        return (
            self.corrupt_images == 0
            and self.corrupt_labels == 0
            and self.invalid_coordinate_boxes == 0
            and self.out_of_range_classes == 0
        )


@dataclass
class ValidationReport:
    """Master validation report across all splits of a domain dataset."""

    domain: str
    dataset_path: str
    allowed_class_ids: List[int]
    allowed_class_names: Dict[int, str]
    splits: Dict[str, SplitValidationResult] = field(default_factory=dict)
    total_images_all_splits: int = 0
    total_labels_all_splits: int = 0
    total_boxes_all_splits: int = 0
    total_pseudo_boxes_all_splits: int = 0
    is_valid_for_training: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, output_path: Union[str, Path]) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=2), encoding="utf-8")
        return p

    def generate_markdown_summary(self) -> str:
        """Generates a human-readable Markdown summary report."""
        lines = [
            f"# Dataset Pre-Flight Validation Report: {self.domain.upper()}",
            f"- **Path:** `{self.dataset_path}`",
            f"- **Allowed Classes:** {self.allowed_class_names}",
            f"- **Overall Training Sanity:** {'PASS' if self.is_valid_for_training else 'FAIL (Contains Fatal Errors)'}",
            f"- **Total Images:** {self.total_images_all_splits}",
            f"- **Total Labels:** {self.total_labels_all_splits}",
            f"- **Total Bounding Boxes:** {self.total_boxes_all_splits}",
            f"- **Pseudo-Boxes Flagged:** {self.total_pseudo_boxes_all_splits}",
            "",
            "## Split Breakdown",
            "",
            "| Split | Images | Labels | Corrupt Imgs | Corrupt Lbls | Invalid Boxes | Pseudo Boxes (%) | Status |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for name, s in self.splits.items():
            status = "PASS" if s.is_clean else "FAIL"
            lines.append(
                f"| `{name}` | {s.total_images} | {s.total_labels} | {s.corrupt_images} | "
                f"{s.corrupt_labels} | {s.invalid_coordinate_boxes} | "
                f"{s.pseudo_boxes} ({s.pseudo_box_percentage}%) | **{status}** |"
            )

        lines.extend(["", "## Class Distribution by Split", ""])
        for name, s in self.splits.items():
            dist_str = ", ".join(f"Class {cls_id} ({self.allowed_class_names.get(cls_id, 'unknown')}): {cnt}"
                                 for cls_id, cnt in sorted(s.class_distribution.items()))
            lines.append(f"- **{name}:** {dist_str if dist_str else 'No annotations'}")

        return "\n".join(lines)


class DatasetValidator:
    """Pre-flight validator checking dataset health, bounding box bounds, and class integrity."""

    def __init__(self, domain: Union[str, Domain]):
        self.domain = domain.value if isinstance(domain, Domain) else str(domain).lower()
        self.local_classes = taxonomy_manager.get_local_classes(self.domain)
        self.allowed_class_ids = set(self.local_classes.keys())

    def validate_image_file(self, img_path: Path) -> Tuple[bool, Optional[str]]:
        """Checks if an image file exists, can be opened, and has valid dimensions."""
        if not img_path.is_file():
            return False, "File does not exist or is not a file"
        if img_path.stat().st_size == 0:
            return False, "Image file is 0 bytes (empty)"
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w, h = img.size
                if w <= 0 or h <= 0:
                    return False, f"Invalid image dimensions: {w}x{h}"
            return True, None
        except Exception as e:
            return False, f"Corrupted image: {str(e)}"

    def validate_label_file(
        self,
        lbl_path: Path,
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        """Validates a single YOLO label file.

        Expected format per line: <class_id> <x_center> <y_center> <width> <height>
        """
        boxes: List[Dict[str, Any]] = []
        errors: List[str] = []

        if not lbl_path.is_file():
            return False, boxes, [f"Label file {lbl_path.name} does not exist"]

        try:
            content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                # Empty file is valid (represents negative background sample)
                return True, boxes, errors

            for line_idx, line in enumerate(content.splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    errors.append(f"Line {line_idx}: expected at least 5 tokens, got {len(parts)}: '{line}'")
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    x_center, y_center, width, height = map(float, parts[1:5])
                except ValueError:
                    errors.append(f"Line {line_idx}: non-numeric values found in '{line}'")
                    continue

                # Coordinate boundary sanity
                invalid_coords = not (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 <= width <= 1.0
                    and 0.0 <= height <= 1.0
                )

                # Pseudo box detection
                is_pseudo = width > 0.95 and height > 0.95

                # Class sanity
                is_out_of_range = cls_id not in self.allowed_class_ids

                boxes.append({
                    "class_id": cls_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "invalid_coords": invalid_coords,
                    "is_pseudo": is_pseudo,
                    "out_of_range": is_out_of_range,
                })

            is_valid = len(errors) == 0
            return is_valid, boxes, errors
        except Exception as e:
            return False, boxes, [f"Failed to read label file: {str(e)}"]

    def validate_split(
        self,
        images_dir: Path,
        labels_dir: Path,
        split_name: str = "split",
        quick_verify_limit: Optional[int] = None,
    ) -> SplitValidationResult:
        """Validates all image/label pairs in a specific split."""
        result = SplitValidationResult(split_name=split_name)

        if not images_dir.exists():
            result.errors.append(f"Images directory missing: {images_dir}")
            return result

        img_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if quick_verify_limit and quick_verify_limit > 0:
            img_files = img_files[:quick_verify_limit]

        result.total_images = len(img_files)

        img_stems = {f.stem: f for f in img_files}
        lbl_stems: Dict[str, Path] = {}

        if labels_dir.exists():
            all_lbl_files = [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
            lbl_stems = {f.stem: f for f in all_lbl_files}
            result.total_labels = len(lbl_stems)

        # 1. Validate images
        for stem, img_path in img_stems.items():
            valid_img, err = self.validate_image_file(img_path)
            if valid_img:
                result.valid_images += 1
            else:
                result.corrupt_images += 1
                result.corrupt_image_files.append(f"{img_path.name}: {err}")
                result.errors.append(f"Image error ({img_path.name}): {err}")

        # 2. Check unpaired items
        for stem in img_stems:
            if stem not in lbl_stems:
                result.images_missing_labels += 1

        for stem in lbl_stems:
            if stem not in img_stems:
                result.orphan_labels += 1

        # 3. Validate labels and bounding boxes
        class_counter = Counter()
        for stem, lbl_path in lbl_stems.items():
            # If quick_verify_limit is set, only evaluate labels corresponding to checked images
            if quick_verify_limit and stem not in img_stems:
                continue

            valid_lbl, boxes, errors = self.validate_label_file(lbl_path)
            if not valid_lbl:
                result.corrupt_labels += 1
                result.corrupt_label_files.append(f"{lbl_path.name}: {'; '.join(errors)}")
                result.errors.extend(errors)
            else:
                result.valid_labels += 1

            for box in boxes:
                result.total_bounding_boxes += 1
                class_counter[box["class_id"]] += 1
                if box["invalid_coords"]:
                    result.invalid_coordinate_boxes += 1
                if box["is_pseudo"]:
                    result.pseudo_boxes += 1
                if box["out_of_range"]:
                    result.out_of_range_classes += 1

        result.class_distribution = dict(class_counter)
        return result

    def validate_dataset(
        self,
        dataset_root: Union[str, Path],
        splits: Optional[List[str]] = None,
        quick_verify_limit: Optional[int] = None,
    ) -> ValidationReport:
        """Validates standard dataset structure with images/<split> and labels/<split>."""
        root = Path(dataset_root)
        splits_to_check = splits or ["train", "val", "test"]

        report = ValidationReport(
            domain=self.domain,
            dataset_path=str(root.resolve()),
            allowed_class_ids=sorted(list(self.allowed_class_ids)),
            allowed_class_names=self.local_classes,
        )

        for split in splits_to_check:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            split_res = self.validate_split(
                images_dir=img_dir,
                labels_dir=lbl_dir,
                split_name=split,
                quick_verify_limit=quick_verify_limit,
            )
            report.splits[split] = split_res
            report.total_images_all_splits += split_res.total_images
            report.total_labels_all_splits += split_res.total_labels
            report.total_boxes_all_splits += split_res.total_bounding_boxes
            report.total_pseudo_boxes_all_splits += split_res.pseudo_boxes

            if not split_res.is_clean:
                report.is_valid_for_training = False

        return report
