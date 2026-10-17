"""Unit tests for Performance Benchmarking Harness (Milestone B6 / REQ-11)."""

from pathlib import Path
from PIL import Image
import pytest

from src.benchmarking.harness import BenchmarkHarness, LatencyPercentiles


def test_latency_percentiles_math():
    # Samples 1 to 100
    samples = list(range(1, 101))
    p = LatencyPercentiles.from_samples(samples)

    assert p.min == 1.0
    assert p.max == 100.0
    assert p.p50 == 50.5
    assert p.p90 == 90.1
    assert p.p95 == 95.05
    assert p.p99 == 99.01
    assert p.mean == 50.5


def test_latency_percentiles_empty():
    p = LatencyPercentiles.from_samples([])
    assert p.mean == 0.0
    assert p.p50 == 0.0


def test_benchmarking_harness_execution(tmp_path: Path):
    # Create 3 synthetic test images
    img_dir = tmp_path / "bench_imgs"
    img_dir.mkdir()
    paths = []
    for i in range(3):
        p = img_dir / f"test_{i}.png"
        img = Image.new("RGB", (640, 640), color="blue")
        img.save(p)
        paths.append(p)

    harness = BenchmarkHarness(device="cpu")
    # Benchmark domain ships for 4 frames with 1 warmup frame
    report = harness.benchmark_domain(
        domain="ships",
        image_paths=paths,
        num_frames=4,
        warmup_frames=1,
    )

    assert report.domain == "ships"
    assert report.total_frames_evaluated == 4
    assert report.warmup_frames == 1
    assert report.fps > 0.0
    assert report.inference_latencies.mean > 0.0
    assert report.total_latencies.p50 > 0.0
    assert report.peak_rss_mb > 0.0

    # Suite markdown generation
    md = harness.generate_suite_report([report])
    assert "# Edge Inference Performance Benchmark Report" in md
    assert "ships" in md
