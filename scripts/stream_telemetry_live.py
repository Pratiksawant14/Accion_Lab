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
    parser.add_argument("--mode", type=str, choices=["auto", "real", "mock"], default="auto", help="Execution mode: 'real' (live YOLO model), 'mock' (simulated edge telemetry), or 'auto' (default)")
    args = parser.parse_args()

    interval_sec = 1.0 / max(0.1, args.fps)

    # Determine execution mode
    use_mock = False
    if args.mode == "mock":
        use_mock = True
    elif args.mode == "auto":
        ckpt_agri = Path("checkpoints/Agriculture_bestweight.pt")
        ckpt_ships = Path("checkpoints/best.pt")
        ckpt_space = Path("checkpoints/space_electronics.pt")
        if not (ckpt_agri.exists() and ckpt_ships.exists() and ckpt_space.exists()):
            use_mock = True
            print("[INFO] Model checkpoints not found in checkpoints/. Defaulting to MOCK simulation mode.")

    print("=" * 80)
    print("TEAM 5B: REAL-TIME EDGE INFERENCE & MQTT EMITTER")
    print("=" * 80)
    print(f"  Broker:     {args.broker}:{args.port}")
    print(f"  Topic:      {args.topic}")
    print(f"  Emission:   {args.fps} FPS (~{interval_sec:.2f}s per frame)")
    print(f"  Mode:       {'MOCK SIMULATION (No weights needed)' if use_mock else 'REAL INFERENCE (YOLO Models Active)'}")
    print(f"  Looping:    {args.loop}")
    print("=" * 80)

    router = None
    if not use_mock:
        # 1. Initialize Inference Router
        print("\n[1/3] Loading Unified Inference Router and Domain Adapters...")
        try:
            router = UnifiedInferenceRouter(lazy_load=False, device="cpu")
            print("      All domain adapters loaded.")
        except Exception as e:
            print(f"      [WARNING] Could not load model adapters ({e}). Falling back to MOCK mode.")
            use_mock = True

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
    streams = {}
    if not use_mock:
        print("\n[3/3] Discovering test images across domains...")
        streams = discover_test_images(limit_per_domain=10)
        total_imgs = sum(len(imgs) for imgs in streams.values())
        print(f"      Loaded {total_imgs} images across {len(streams)} domains.")
        if total_imgs == 0:
            print("      [WARNING] No test images found in Synthline. Switching to MOCK mode.")
            use_mock = True
    else:
        print("\n[3/3] Running in MOCK mode (simulated edge video feed)...")

    print("\n" + "=" * 80)
    print("EMITTING REAL-TIME INFERENCE TELEMETRY (Press Ctrl+C to stop)...")
    print("=" * 80)

    emitted_count = 0
    cycle = 0

    # Mock scenarios for standalone testing without heavy datasets
    mock_scenarios = [
        {"domain": "agriculture", "model": "agriculture_yolov8n", "latency": 48.5, "boxes": [[0.0, 0.35, 0.99, 0.99]], "labels": ["healthy_crop"], "ids": [0], "confs": [0.97], "pseudos": [false]},
        {"domain": "agriculture", "model": "agriculture_yolov8n", "latency": 35.2, "boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": ["healthy_crop"], "ids": [0], "confs": [0.99], "pseudos": [true]},
        {"domain": "ships", "model": "ships_yolo26n", "latency": 52.1, "boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": ["cargo_ship"], "ids": [2], "confs": [0.98], "pseudos": [true]},
        {"domain": "ships", "model": "ships_yolo26n", "latency": 41.3, "boxes": [[0.12, 0.25, 0.65, 0.78]], "labels": ["passenger_ship"], "ids": [3], "confs": [0.89], "pseudos": [false]},
        {"domain": "space_electronics", "model": "space_yolo26n", "latency": 49.8, "boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": ["short"], "ids": [5], "confs": [0.95], "pseudos": [true]},
        {"domain": "space_electronics", "model": "space_yolo26n", "latency": 39.4, "boxes": [[0.22, 0.18, 0.54, 0.62]], "labels": ["open"], "ids": [4], "confs": [0.91], "pseudos": [false]},
    ]
    # Replace json booleans
    for s in mock_scenarios:
        s["pseudos"] = [bool(p) for p in s["pseudos"]]

    try:
        while True:
            cycle += 1
            if use_mock:
                for idx, scen in enumerate(mock_scenarios):
                    frame_start = time.perf_counter()
                    emitted_count += 1
                    frame_id = f"frame_{scen['domain']}_c{cycle}_{idx+1:03d}"

                    payload = {
                        "frame_id": frame_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "domain": scen["domain"],
                        "model_name": scen["model"],
                        "model_version": "1.0.0",
                        "latency_ms": scen["latency"],
                        "bounding_boxes": scen["boxes"],
                        "class_labels": scen["labels"],
                        "global_class_ids": scen["ids"],
                        "confidence_scores": scen["confs"],
                        "pseudo_box_flags": scen["pseudos"],
                    }

                    publisher.publish(payload)
                    pub_status = "SENT" if publisher.get_metrics()["connection_state"] == "CONNECTED" else "BUFFERED"

                    det_count = len(scen["labels"])
                    pseudo_count = sum(1 for p in scen["pseudos"] if p)
                    det_preview = f"{scen['labels'][0]} ({scen['confs'][0]:.2f})"

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] #{emitted_count:04d} "
                        f"| {scen['domain'].upper():<17} "
                        f"| {frame_id:<22} "
                        f"| Lat: {scen['latency']:5.1f}ms "
                        f"| Dets: {det_count} (Pseudo: {pseudo_count}) "
                        f"| [{pub_status}] {det_preview}"
                    )

                    if args.max_frames > 0 and emitted_count >= args.max_frames:
                        print(f"\nReached max frames limit ({args.max_frames}). Stopping.")
                        return

                    elapsed = time.perf_counter() - frame_start
                    time.sleep(max(0.0, interval_sec - elapsed))
            else:
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

                        # Publish canonical telemetry payload to MQTT
                        publisher.publish(result)
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
                        time.sleep(max(0.0, interval_sec - elapsed))

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\nEmission stopped by user.")
    finally:
        publisher.stop()
        print("MQTT Publisher stopped cleanly.")


if __name__ == "__main__":
    main()
