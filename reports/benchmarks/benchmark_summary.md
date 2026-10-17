# Edge Inference Performance Benchmark Report
**Execution Host:** Local Edge Simulation (CPU)  
**Requirement Verification:** REQ-11 & REQ-09 Latency Target  

## Summary Table across Domains

| Domain | Model Variant | Frames | FPS | Inf p50 (ms) | Inf p95 (ms) | Inf p99 (ms) | Total p50 (ms) | Total p95 (ms) | RAM Peak (MB) | <=50ms Target |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `agriculture` | `agriculture_yolov8n` | 100 | **5.39** | 147.54 | 179.39 | 254.6 | 181.37 | 215.66 | 399.44 | **REVIEW (CPU)** |
| `ships` | `ships_yolo26n` | 100 | **4.22** | 192.53 | 231.27 | 337.69 | 230.47 | 270.38 | 421.63 | **REVIEW (CPU)** |
| `space_electronics` | `space_electronics_yolo26n` | 100 | **4.24** | 191.09 | 240.95 | 275.02 | 230.26 | 287.19 | 442.94 | **REVIEW (CPU)** |

## Detailed Timing Breakdown (p50 / p95 / p99 in ms)

### Domain: AGRICULTURE (agriculture_yolov8n)
- **Throughput:** 5.39 FPS (100 frames evaluated in 18.566s)
- **Preprocessing:** Mean=24.56ms, p50=24.81ms, p95=27.49ms, p99=31.61ms
- **Model Forward Pass:** Mean=152.19ms, p50=147.54ms, p95=179.39ms, p99=254.6ms
- **Postprocessing / NMS:** Mean=0.12ms, p50=0.12ms, p95=0.16ms, p99=0.26ms
- **End-to-End Latency:** Mean=185.54ms, p50=181.37ms, p95=215.66ms, p99=305.43ms
- **Memory Usage:** Initial=263.38 MB, Peak=399.44 MB, Final=398.16 MB
- **Detections:** 109 objects detected (105 flagged as full-frame pseudo-boxes)

### Domain: SHIPS (ships_yolo26n)
- **Throughput:** 4.22 FPS (100 frames evaluated in 23.681s)
- **Preprocessing:** Mean=23.71ms, p50=24.13ms, p95=27.58ms, p99=28.26ms
- **Model Forward Pass:** Mean=198.06ms, p50=192.53ms, p95=231.27ms, p99=337.69ms
- **Postprocessing / NMS:** Mean=0.12ms, p50=0.12ms, p95=0.17ms, p99=0.19ms
- **End-to-End Latency:** Mean=236.69ms, p50=230.47ms, p95=270.38ms, p99=383.02ms
- **Memory Usage:** Initial=398.9 MB, Peak=421.63 MB, Final=411.5 MB
- **Detections:** 100 objects detected (100 flagged as full-frame pseudo-boxes)

### Domain: SPACE_ELECTRONICS (space_electronics_yolo26n)
- **Throughput:** 4.24 FPS (100 frames evaluated in 23.608s)
- **Preprocessing:** Mean=25.5ms, p50=25.14ms, p95=28.52ms, p99=41.64ms
- **Model Forward Pass:** Mean=196.09ms, p50=191.09ms, p95=240.95ms, p99=275.02ms
- **Postprocessing / NMS:** Mean=0.13ms, p50=0.13ms, p95=0.17ms, p99=0.25ms
- **End-to-End Latency:** Mean=235.97ms, p50=230.26ms, p95=287.19ms, p99=322.91ms
- **Memory Usage:** Initial=411.52 MB, Peak=442.94 MB, Final=438.3 MB
- **Detections:** 112 objects detected (112 flagged as full-frame pseudo-boxes)
