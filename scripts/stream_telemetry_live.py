"""Continuous Real-Time Edge Inference & MQTT Telemetry Emitter.

Simulates physical edge cameras streaming multi-domain frames (Agriculture, Ships,
Space Electronics), executing YOLO/ONNX inference, mapping global canonical taxonomy,
and publishing real-time telemetry over MQTT to Team 5C.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.router import UnifiedInferenceRouter
from src.inference.schema import Domain, InferenceRequest, InferenceResult
from src.telemetry.mqtt_publisher import MQTTTelemetryPublisher, TelemetryConfig


def discover_test_images(limit_per_domain: int = 15) -> Dict[Domain, List[Path]]:
    """Discovers test images from Synthline datasets."""
    domain_dirs = {
        Domain.AGRICULTURE: Path("Synthline/agriculture robotics/agriculture_final_dataset/images/test"),
        Domain.SHIPS: Path("Synthline/environmental defense/ship_final_dataset/images/test"),
        Domain.SPACE_ELECTRONICS: Path("Synthline/space electronics/space_final_dataset/images/test"),
    }
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    streams: Dict[Domain, List[Path]] = {}
    for dom, folder in domain_dirs.items():
        found = []
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in exts:
                    found.append(p)
                    if len(found) >= limit_per_domain:
                        break
        streams[dom] = found
    return streams


def main():
    parser = argparse.ArgumentParser(description="Team 5B Live Real-Time MQTT Telemetry Emitter")
    parser.add_argument("--broker", type=str, default="broker.hivemq.com", help="MQTT broker host (default: broker.hivemq.com)")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    parser.add_argument("--topic", type=str, default="telemetry/edge/team5b/inference", help="MQTT topic")
    parser.add_argument("--fps", type=float, default=1.0, help="Target emission rate in FPS (default: 1.0)")
    parser.add_argument("--loop", action="store_true", default=True, help="Continuously loop dataset frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to emit (0 = infinite)")
    args = parser.parse_args()

    interval_sec = 1.0 / max(0.1, args.fps)

    print("=" * 80)
    print("TEAM 5B: REAL-TIME EDGE INFERENCE & MQTT EMITTER")
    print("=" * 80)
    print(f"  Broker:     {args.broker}:{args.port}")
    print(f"  Topic:      {args.topic}")
    print(f"  Emission:   {args.fps} FPS (~{interval_sec:.2f}s per frame)")
    print(f"  Looping:    {args.loop}")
    print("=" * 80)

    # 1. Initialize Inference Router
    print("\n[1/3] Loading Unified Inference Router and Domain Adapters...")
    router = UnifiedInferenceRouter(lazy_load=False, device="cpu")
    print("      All domain adapters loaded.")

    # 2. Initialize Resilient MQTT Publisher
    print(f"\n[2/3] Connecting to MQTT Broker at {args.broker}:{args.port}...")
    telemetry_cfg = TelemetryConfig(
        broker_host=args.broker,
        broker_port=args.port,
        topic=args.topic,
        buffer_capacity=1000,
    )
    publisher = MQTTTelemetryPublisher(telemetry_cfg)
    publisher.start()
    time.sleep(1.0)  # Allow MQTT connect thread to establish link

    metrics = publisher.get_metrics()
    print(f"      MQTT Connection State: {metrics['connection_state']}")

    # 3. Discover Dataset Frames
    print("\n[3/3] Discovering test images across domains...")
    streams = discover_test_images(limit_per_domain=10)
    total_imgs = sum(len(imgs) for imgs in streams.values())
    print(f"      Loaded {total_imgs} images across {len(streams)} domains.")

    print("\n" + "=" * 80)
    print("EMITTING REAL-TIME INFERENCE TELEMETRY (Press Ctrl+C to stop)...")
    print("=" * 80)

    emitted_count = 0
    cycle = 0

    try:
        while True:
            cycle += 1
            for dom, img_list in streams.items():
                for idx, img_path in enumerate(img_list):
                    frame_start = time.perf_counter()
                    emitted_count += 1
                    frame_id = f"frame_{dom.value}_c{cycle}_{idx+1:03d}"

                    # Run inference through Unified Router
                    req = InferenceRequest(
                        image=str(img_path),
                        domain=dom,
                        frame_id=frame_id,
                        confidence_threshold=0.25,
                    )
                    result: InferenceResult = router.predict(req)
                    inference_elapsed_ms = (time.perf_counter() - frame_start) * 1000.0

                    # Publish canonical telemetry payload to MQTT
                    published = publisher.publish(result)
                    pub_status = "SENT" if publisher.get_metrics()["connection_state"] == "CONNECTED" else "BUFFERED"

                    # Format terminal log
                    det_count = len(result.detections)
                    pseudo_count = sum(1 for d in result.detections if d.box.is_pseudo_box)
                    det_labels = [f"{d.global_class_name} ({d.confidence:.2f})" for d in result.detections[:3]]
                    det_preview = ", ".join(det_labels) if det_labels else "No detections"

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] #{emitted_count:04d} "
                        f"| {dom.value.upper():<17} "
                        f"| {frame_id:<22} "
                        f"| Inf: {result.inference_time_ms:5.1f}ms "
                        f"| Dets: {det_count} (Pseudo: {pseudo_count}) "
                        f"| [{pub_status}] {det_preview}"
                    )

                    if args.max_frames > 0 and emitted_count >= args.max_frames:
                        print(f"\nReached max frames limit ({args.max_frames}). Stopping.")
                        return

                    # Regulate frame rate
                    elapsed = time.perf_counter() - frame_start
                    sleep_needed = max(0.0, interval_sec - elapsed)
                    time.sleep(sleep_needed)

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\nEmission stopped by user.")
    finally:
        publisher.stop()
        print("MQTT Publisher stopped cleanly.")


if __name__ == "__main__":
    main()
