"""CLI Runner to execute official Milestone B6 benchmarks across all three domains."""

import json
import logging
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.harness import BenchmarkHarness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkRunner")


def run_all_benchmarks(num_frames: int = 100, warmup_frames: int = 5):
    logger.info("Initializing Unified Benchmark Harness on CPU...")
    harness = BenchmarkHarness(device="cpu")

    domain_configs = [
        {
            "domain": "agriculture",
            "img_dir": REPO_ROOT / "Synthline" / "agriculture robotics" / "agriculture_final_dataset" / "images" / "val",
        },
        {
            "domain": "ships",
            "img_dir": REPO_ROOT / "Synthline" / "environmental defense" / "ship_final_dataset" / "images" / "val",
        },
        {
            "domain": "space_electronics",
            "img_dir": REPO_ROOT / "Synthline" / "space electronics" / "space_final_dataset" / "images" / "val",
        },
    ]

    reports = []
    output_dir = REPO_ROOT / "reports" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in domain_configs:
        domain = cfg["domain"]
        img_dir = cfg["img_dir"]
        logger.info("Scanning image pool for %s in %s...", domain, img_dir)
        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not img_files:
            logger.error("No images found for domain %s in %s!", domain, img_dir)
            continue

        logger.info("Found %d images for %s. Benchmarking %d frames...", len(img_files), domain, num_frames)
        report = harness.benchmark_domain(
            domain=domain,
            image_paths=img_files,
            num_frames=num_frames,
            warmup_frames=warmup_frames,
        )
        reports.append(report)

        # Save individual domain report
        domain_json_path = output_dir / f"{domain}_benchmark.json"
        domain_json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Domain %s benchmark completed: FPS=%.2f, Inf p50=%.2fms", domain, report.fps, report.inference_latencies.p50)

    # Generate suite-wide Markdown and JSON summaries
    suite_md = harness.generate_suite_report(reports)
    suite_md_path = output_dir / "benchmark_summary.md"
    suite_md_path.write_text(suite_md, encoding="utf-8")

    suite_json_data = [r.to_dict() for r in reports]
    suite_json_path = output_dir / "benchmark_summary.json"
    suite_json_path.write_text(json.dumps(suite_json_data, indent=2), encoding="utf-8")

    logger.info("=======================================================")
    logger.info("BENCHMARK SUITE FINISHED SUCCESSFULLY!")
    logger.info("Reports saved to:")
    logger.info("  - %s", suite_md_path)
    logger.info("  - %s", suite_json_path)
    logger.info("=======================================================")
    print("\n" + suite_md)


if __name__ == "__main__":
    frames = 100
    if len(sys.argv) > 1:
        frames = int(sys.argv[1])
    run_all_benchmarks(num_frames=frames)
