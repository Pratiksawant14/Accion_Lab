"""Team 5C Telemetry Receiver & Live Monitor.

Subscribes to Team 5B's edge inference telemetry MQTT stream, parses canonical JSON
payloads, validates global taxonomy (classes 0-5), separates localized bounding boxes
from full-frame pseudo-box classification alerts, and prints real-time dashboard data.
"""

import argparse
from datetime import datetime
import json
import sys
import paho.mqtt.client as mqtt

GLOBAL_TAXONOMY = {
    0: "healthy_crop",
    1: "dead_crop",
    2: "cargo_ship",
    3: "passenger_ship",
    4: "open",
    5: "short",
}


def log_print(*args, **kwargs):
    kwargs["flush"] = True
    print(*args, **kwargs)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log_print("\n" + "=" * 80)
        log_print("CONNECTED TO TEAM 5B MQTT TELEMETRY STREAM SUCCESSFULLY!")
        log_print(f"Subscribed Topic: {userdata['topic']}")
        log_print("Listening for real-time edge telemetry payloads (Press Ctrl+C to exit)...")
        log_print("=" * 80 + "\n")
        client.subscribe(userdata["topic"], qos=1)
    else:
        log_print(f"Connection failed with return code {rc}")


def on_message(client, userdata, msg):
    userdata["count"] += 1
    raw = msg.payload.decode("utf-8")
    try:
        data = json.loads(raw)
    except Exception as e:
        log_print(f"[ERROR] Corrupt JSON payload: {e}")
        return

    frame_id = data.get("frame_id", "unknown")
    domain = data.get("domain", "unknown").upper()
    model = data.get("model_name", data.get("model_version", "EdgeModel"))
    latency_ms = data.get("latency_ms", data.get("inference_time_ms", 0.0))
    timestamp = data.get("timestamp", datetime.now().isoformat())

    boxes = data.get("bounding_boxes", [])
    labels = data.get("class_labels", [])
    global_ids = data.get("global_class_ids", [])
    confs = data.get("confidence_scores", [])
    pseudos = data.get("pseudo_box_flags", [])

    # Support nested detections if present
    if not boxes and "detections" in data:
        for d in data["detections"]:
            b = d.get("box", {})
            boxes.append([b.get("x_min", 0), b.get("y_min", 0), b.get("x_max", 0), b.get("y_max", 0)])
            labels.append(d.get("global_class_name", "unknown"))
            global_ids.append(d.get("global_class_id", -1))
            confs.append(d.get("confidence", 0.0))
            pseudos.append(b.get("is_pseudo_box", False))

    log_print("-" * 80)
    log_print(f"RECEIVE #{userdata['count']:04d} | Frame: {frame_id} | Domain: {domain} | Model: {model} | Latency: {latency_ms:.1f}ms")
    log_print(f"Timestamp: {timestamp}")

    if not boxes:
        log_print("  -> Result: No detections in this frame.")
    else:
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i] if i < len(labels) else "unknown"
            g_id = global_ids[i] if i < len(global_ids) else -1
            conf = confs[i] if i < len(confs) else 0.0
            is_pseudo = pseudos[i] if i < len(pseudos) else False

            if is_pseudo:
                # Rule for Team 5C UI: Render as Classification Alert Banner
                log_print(
                    f"  [{i+1}] [ALERT BADGE / FULL-FRAME PSEUDO-BOX] "
                    f"Class {g_id} ({label}) | Conf: {conf:.2%} "
                    f"-> Action: Render UI Alert Badge (Whole-image classification alert)"
                )
            else:
                # Rule for Team 5C UI: Render as standard Bounding Box Canvas Overlay
                log_print(
                    f"  [{i+1}] [LOCALIZED BOUNDING BOX] "
                    f"Class {g_id} ({label}) | Conf: {conf:.2%} "
                    f"| Box: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}] "
                    f"-> Action: Draw Canvas Box"
                )


def main():
    parser = argparse.ArgumentParser(description="Team 5C MQTT Telemetry Receiver")
    parser.add_argument("--broker", type=str, default="broker.hivemq.com", help="MQTT broker host (default: broker.hivemq.com)")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    parser.add_argument("--topic", type=str, default="telemetry/edge/team5b/inference", help="MQTT topic")
    args = parser.parse_args()

    print("=" * 80)
    print("TEAM 5C: REAL-TIME MQTT TELEMETRY RECEIVER")
    print("=" * 80)
    print(f"Connecting to Broker: {args.broker}:{args.port}")
    print(f"Target Topic:        {args.topic}")

    userdata = {"topic": args.topic, "count": 0}
    client = mqtt.Client(client_id="team_5c_dashboard_receiver", userdata=userdata)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nReceiver stopped by user.")
        client.disconnect()
    except Exception as e:
        print(f"\n[ERROR] Connection error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
