"""CLI script to export all 3 domain checkpoints to ONNX and verify numerical equivalence."""

import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.export.onnx_exporter import ONNXExporter
from src.export.equivalence import OutputEquivalenceValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportRunner")


def main():
    exports_dir = REPO_ROOT / "artifacts" / "exports"
    reports_dir = REPO_ROOT / "reports" / "exports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    exporter = ONNXExporter(exports_dir=exports_dir)
    validator = OutputEquivalenceValidator(tolerance=1e-3)

    configs = [
        {
            "domain": "agriculture",
            "checkpoint": REPO_ROOT / "checkpoints" / "Agriculture_bestweight.pt",
            "imgsz": 512,
            "target_filename": "agriculture.onnx",
            "val_images": list((REPO_ROOT / "Synthline" / "agriculture robotics" / "agriculture_final_dataset" / "images" / "val").glob("*.png"))[:5],
        },
        {
            "domain": "ships",
            "checkpoint": REPO_ROOT / "checkpoints" / "best.pt",
            "imgsz": 640,
            "target_filename": "ships.onnx",
            "val_images": list((REPO_ROOT / "Synthline" / "environmental defense" / "ship_final_dataset" / "images" / "val").glob("*.png"))[:5],
        },
        {
            "domain": "space_electronics",
            "checkpoint": REPO_ROOT / "checkpoints" / "space_electronics.pt",
            "imgsz": 640,
            "target_filename": "space_electronics.onnx",
            "val_images": list((REPO_ROOT / "Synthline" / "space electronics" / "space_final_dataset" / "images" / "val").glob("*.png"))[:5],
        },
    ]

    export_results = []
    equivalence_reports = []

    for cfg in configs:
        domain = cfg["domain"]
        ckpt = cfg["checkpoint"]
        imgsz = cfg["imgsz"]
        target_name = cfg["target_filename"]
        val_imgs = cfg["val_images"]

        logger.info("=======================================================")
        logger.info("STEP 1: Exporting %s to ONNX (%s, imgsz=%d)...", domain.upper(), ckpt.name, imgsz)
        logger.info("=======================================================")

        res = exporter.export(
            checkpoint_path=ckpt,
            domain=domain,
            target_filename=target_name,
            imgsz=imgsz,
            opset=12,
        )
        export_results.append(res.to_dict())

        logger.info("=======================================================")
        logger.info("STEP 2: Verifying Numerical Equivalence for %s...", domain.upper())
        logger.info("=======================================================")

        eq_report = validator.verify(
            domain=domain,
            checkpoint_path=ckpt,
            onnx_path=res.onnx_path,
            imgsz=imgsz,
            sample_image_paths=val_imgs,
            num_random_samples=5,
        )
        equivalence_reports.append(eq_report)

        logger.info(
            "Equivalence Result for %s: Status=%s (Max Abs Error: %.4e, Mean: %.4e)",
            domain,
            "PASS" if eq_report.all_passed else "FAIL",
            eq_report.max_absolute_error_overall,
            eq_report.mean_absolute_error_overall,
        )

    # Save summary reports
    export_json = reports_dir / "onnx_export_summary.json"
    export_json.write_text(json.dumps(export_results, indent=2), encoding="utf-8")

    eq_json = reports_dir / "onnx_equivalence_summary.json"
    eq_json.write_text(json.dumps([r.to_dict() for r in equivalence_reports], indent=2), encoding="utf-8")

    combined_md = ["# Combined ONNX Export & Output Equivalence Summary\n"]
    for r in equivalence_reports:
        combined_md.append(r.generate_markdown_summary() + "\n---\n")

    eq_md = reports_dir / "onnx_equivalence_summary.md"
    eq_md.write_text("\n".join(combined_md), encoding="utf-8")

    logger.info("=======================================================")
    logger.info("ALL EXPORTS & EQUIVALENCE VALIDATIONS COMPLETED!")
    logger.info("Reports saved to: %s", reports_dir)
    logger.info("=======================================================")


if __name__ == "__main__":
    main()
