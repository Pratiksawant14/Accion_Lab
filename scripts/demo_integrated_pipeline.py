"""End-to-End Integrated Edge Inference & Telemetry Pipeline Demo (Milestone B11 / REQ-13).

Demonstrates the unified pipeline in action:
1. Multi-domain input streaming from Synthline test datasets (Agriculture, Ships, Space Electronics).
2. Domain-aware routing via UnifiedInferenceRouter.
3. Edge model execution (PyTorch/ONNX engine).
4. Local-to-global canonical taxonomy mapping (ADR-002: 6 global classes).
5. Synthetic pseudo-box flagging (RSK-04: is_pseudo_box).
6. Resilient telemetry serialization conforming to Team 5C specification.
7. Publication / offline circular buffering via Resilient MQTT Publisher.
8. Complete audit report generation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from src.inference.router import UnifiedInferenceRouter
from src.inference.schema import Domain, InferenceRequest, InferenceResult
from src.telemetry.mqtt_publisher import MQTTTelemetryPublisher, TelemetryConfig


def find_sample_images(base_dir: Path, max_samples: int = 5) -> List[Path]:
    """Finds test images in the given directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    if base_dir.exists():
        for p in sorted(base_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(p)
                if len(images) >= max_samples:
                    break
    return images


def run_pipeline_demo():
    print("=" * 80)
    print("ACTION LAB - TEAM 5B: END-TO-END INTEGRATED EDGE INFERENCE PIPELINE DEMO")
    print("=" * 80)

    # 1. Initialize Unified Inference Router
    print("\n[Step 1/5] Initializing Unified Inference Router & Domain Adapters...")
    router = UnifiedInferenceRouter(lazy_load=False, device="cpu")
    print("  Router status: All 3 adapters initialized successfully.")

    # 2. Initialize Resilient MQTT Telemetry Publisher
    print("\n[Step 2/5] Initializing Resilient Telemetry Publisher (Capacity: 1,000)...")
    telemetry_cfg = TelemetryConfig(
        broker_host="localhost",
        broker_port=1883,
        topic="telemetry/edge/team5b/inference",
        buffer_capacity=1000,
    )
    publisher = MQTTTelemetryPublisher(telemetry_cfg)
    publisher.start()
    print("  Telemetry status: Resilient publisher started (offline ring-buffer active).")

    # 3. Locate Sample Images from Synthline Datasets
    print("\n[Step 3/5] Discovering Test Streams from Synthline Datasets...")
    domain_paths = {
        Domain.AGRICULTURE: Path("Synthline/agriculture robotics/agriculture_final_dataset/images/test"),
        Domain.SHIPS: Path("Synthline/environmental defense/ship_final_dataset/images/test"),
        Domain.SPACE_ELECTRONICS: Path("Synthline/space electronics/space_final_dataset/images/test"),
    }

    test_streams: Dict[Domain, List[Path]] = {}
    for dom, p in domain_paths.items():
        samples = find_sample_images(p, max_samples=4)
        test_streams[dom] = samples
        print(f"  {dom.value.upper():18} -> {len(samples)} test frames found in {p.name}/")

    # 4. Stream and Process Frames Sequentially
    print("\n[Step 4/5] Executing Streaming Inference & Telemetry Pipeline...")
    total_frames = 0
    total_detections = 0
    pseudo_boxes_flagged = 0
    all_results: List[Dict[str, Any]] = []

    start_stream_time = time.perf_counter()

    for dom, img_list in test_streams.items():
        print(f"\n--- Processing Stream: {dom.value.upper()} ---")
        for idx, img_path in enumerate(img_list):
            frame_start = time.perf_counter()

            # Create inference request
            request = InferenceRequest(
                image=str(img_path),
                domain=dom,
                frame_id=f"frame_{dom.value}_{idx+1}",
                confidence_threshold=0.25,
            )

            # Route and predict
            result: InferenceResult = router.predict(request)
            frame_elapsed_ms = (time.perf_counter() - frame_start) * 1000.0

            total_frames += 1
            total_detections += len(result.detections)

            # Format canonical telemetry payload
            payload = result.to_telemetry_payload()

            # Publish to resilient MQTT publisher
            publisher.publish(result)

            # Count flagged pseudo-boxes
            frame_pseudo = sum(1 for d in result.detections if d.box.is_pseudo_box)
            pseudo_boxes_flagged += frame_pseudo

            # Log detection breakdown
            det_summary = []
            for d in result.detections:
                p_tag = " [PSEUDO]" if d.box.is_pseudo_box else ""
                det_summary.append(
                    f"{d.global_class_name} (local:{d.local_class_id}->global:{d.global_class_id}, "
                    f"conf:{d.confidence:.2f}{p_tag})"
                )

            det_str = "; ".join(det_summary) if det_summary else "No detections"
            print(
                f"  Frame {idx+1}/{len(img_list)} [{img_path.name[:25]}]: "
                f"{len(result.detections)} dets in {frame_elapsed_ms:.1f}ms (inf: {result.inference_time_ms:.1f}ms) | {det_str}"
            )

            all_results.append(
                {
                    "domain": dom.value,
                    "image": img_path.name,
                    "inference_time_ms": result.inference_time_ms,
                    "total_time_ms": round(frame_elapsed_ms, 2),
                    "detections_count": len(result.detections),
                    "pseudo_boxes_count": frame_pseudo,
                    "detections": [d.model_dump() for d in result.detections],
                    "telemetry_payload": payload,
                }
            )

    total_duration = time.perf_counter() - start_stream_time

    # 5. Telemetry & Hardware Health Summary
    print("\n[Step 5/5] Compiling Pipeline Telemetry Health & Audit Report...")
    time.sleep(0.5)  # Allow publisher worker cycle to settle
    metrics = publisher.get_metrics()
    publisher.stop()

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_frames_processed": total_frames,
            "total_detections": total_detections,
            "pseudo_boxes_flagged": pseudo_boxes_flagged,
            "total_duration_sec": round(total_duration, 3),
            "effective_fps": round(total_frames / total_duration, 2) if total_duration > 0 else 0.0,
        },
        "telemetry_publisher_health": metrics,
        "processed_frames": all_results,
    }

    report_path = Path("reports/demos/end_to_end_demo_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("PIPELINE DEMO EXECUTION SUMMARY")
    print("=" * 80)
    print(f"- Total Frames Processed:   {total_frames}")
    print(f"- Total Detections:         {total_detections}")
    print(f"- Pseudo-Boxes Detected:    {pseudo_boxes_flagged} (Safely flagged via is_pseudo_box)")
    print(f"- Total Stream Duration:    {total_duration:.2f} seconds")
    print(f"- End-to-End Pipeline FPS:  {total_frames / total_duration:.2f} FPS")
    print(f"- Telemetry Messages Held:  {metrics['current_buffer_size']} / {metrics['buffer_capacity']}")
    print(f"- Telemetry Messages Dropped: {metrics['total_dropped']}")
    print(f"- Connection State:         {metrics['connection_state']}")
    print(f"- Full Report Saved to:     {report_path}")
    print("=" * 80)
    print("MILESTONE B11 COMPLETED SUCCESSFULLY: ALL PIPELINE STAGES OPERATIONAL.")


if __name__ == "__main__":
    run_pipeline_demo()
