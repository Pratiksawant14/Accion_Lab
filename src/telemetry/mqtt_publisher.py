"""Resilient Edge MQTT Telemetry Publisher for Team 5B.

Conforms to Requirements REQ-09 and REQ-10 (Milestone B9):
- Publishes canonical inference telemetry payloads according to Team 5C specification.
- Features a thread-safe circular memory buffer (capacity >= 1,000 messages) during network disruptions.
- Implements exponential backoff reconnection with random jitter.
- Triggers explicit system alerting when disconnection lasts >= 60 seconds.
- Automatically flushes queued messages in FIFO order upon network restoration.
- Provides complete telemetry health metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
import paho.mqtt.client as mqtt

from ..inference.schema import InferenceResult

logger = logging.getLogger("TelemetryPublisher")


class ConnectionState(str, Enum):
    CONNECTED = "CONNECTED"
    CONNECTING = "CONNECTING"
    DISCONNECTED = "DISCONNECTED"
    DEGRADED_BUFFERING = "DEGRADED_BUFFERING"


class BufferOverflowPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"  # Standard FIFO ring buffer
    DROP_NEWEST = "drop_newest"


@dataclass
class TelemetryConfig:
    """Configuration for MQTT telemetry client and buffering policies."""

    broker_host: str = "localhost"
    broker_port: int = 1883
    topic: str = "telemetry/edge/team5b/inference"
    qos: int = 1
    keepalive_sec: int = 60
    client_id: Optional[str] = None

    # Resiliency parameters (conforming to REQ-10)
    buffer_capacity: int = 1000
    overflow_policy: BufferOverflowPolicy = BufferOverflowPolicy.DROP_OLDEST
    backoff_initial_sec: float = 1.0
    backoff_factor: float = 2.0
    backoff_max_sec: float = 30.0
    backoff_jitter: float = 0.2  # +/- 20% random jitter
    failure_warning_threshold_sec: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MQTTTelemetryPublisher:
    """Thread-safe resilient MQTT publisher with circular buffer and auto-reconnect."""

    def __init__(
        self,
        config: Optional[TelemetryConfig] = None,
        on_outage_warning_callback: Optional[Callable[[float], None]] = None,
    ):
        self.config = config or TelemetryConfig()
        self.on_outage_warning_callback = on_outage_warning_callback

        self._state = ConnectionState.DISCONNECTED
        self._buffer: deque = deque(maxlen=self.config.buffer_capacity)
        self._buffer_lock = threading.Lock()

        # Telemetry metrics counters
        self._total_published = 0
        self._total_buffered = 0
        self._total_dropped = 0
        self._last_connected_time: Optional[float] = None
        self._disconnected_start_time: Optional[float] = time.time()
        self._outage_warning_triggered = False

        # Reconnection control
        self._current_backoff = self.config.backoff_initial_sec
        self._stop_event = threading.Event()
        self._reconnect_thread: Optional[threading.Thread] = None

        # Paho MQTT Client setup
        client_id = self.config.client_id or f"team5b_edge_{int(time.time())}"
        self._client = mqtt.Client(client_id=client_id, clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def buffer_size(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    @property
    def outage_duration_sec(self) -> float:
        if self._state == ConnectionState.CONNECTED:
            return 0.0
        if self._disconnected_start_time is None:
            return 0.0
        return max(0.0, time.time() - self._disconnected_start_time)

    def start(self, background_loop: bool = True) -> None:
        """Starts the MQTT client and launches the background reconnection manager."""
        self._stop_event.clear()
        self._trigger_connect()

        if background_loop:
            self._client.loop_start()

        # Start dedicated reconnection manager
        self._reconnect_thread = threading.Thread(
            target=self._reconnection_loop,
            name="MQTT-Reconnect-Worker",
            daemon=True,
        )
        self._reconnect_thread.start()
        logger.info("MQTT Telemetry Publisher started. Target: %s:%d (Topic: %s)",
                    self.config.broker_host, self.config.broker_port, self.config.topic)

    def stop(self) -> None:
        """Gracefully shuts down publisher and background threads."""
        self._stop_event.set()
        try:
            self._client.disconnect()
            self._client.loop_stop()
        except Exception:
            pass
        self._state = ConnectionState.DISCONNECTED
        logger.info("MQTT Telemetry Publisher stopped.")

    def _trigger_connect(self) -> bool:
        """Attempts an immediate network socket connection to the broker."""
        self._state = ConnectionState.CONNECTING
        try:
            self._client.connect(
                host=self.config.broker_host,
                port=self.config.broker_port,
                keepalive=self.config.keepalive_sec,
            )
            return True
        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            logger.debug("Connection attempt to %s:%d failed: %s", self.config.broker_host, self.config.broker_port, e)
            return False

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            logger.info("Successfully connected to MQTT broker %s:%d", self.config.broker_host, self.config.broker_port)
            self._state = ConnectionState.CONNECTED
            self._last_connected_time = time.time()
            self._disconnected_start_time = None
            self._outage_warning_triggered = False
            self._current_backoff = self.config.backoff_initial_sec

            # Flush queued buffer messages in FIFO order
            self._flush_buffer()
        else:
            logger.warning("MQTT connection refused with result code: %d", rc)
            self._state = ConnectionState.DISCONNECTED

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        logger.warning("Disconnected from MQTT broker (rc=%d). Entering DEGRADED_BUFFERING state.", rc)
        self._state = ConnectionState.DEGRADED_BUFFERING
        if self._disconnected_start_time is None:
            self._disconnected_start_time = time.time()

    def publish(
        self,
        data: Union[InferenceResult, Dict[str, Any], str],
        topic: Optional[str] = None,
    ) -> bool:
        """Publishes a telemetry payload or queues it in the buffer if offline."""
        target_topic = topic or self.config.topic

        # Normalize to JSON string
        if isinstance(data, InferenceResult):
            payload_dict = data.to_telemetry_payload()
            payload_str = json.dumps(payload_dict)
        elif isinstance(data, dict):
            payload_str = json.dumps(data)
        elif isinstance(data, str):
            payload_str = data
        else:
            payload_str = json.dumps({"data": str(data)})

        # Check for sustained network outage warning condition (>= 60s)
        self._check_outage_warning()

        # If connected, attempt immediate network dispatch
        if self._state == ConnectionState.CONNECTED:
            try:
                info = self._client.publish(target_topic, payload_str, qos=self.config.qos)
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    self._total_published += 1
                    return True
                else:
                    logger.debug("Client.publish failed with rc=%d. Enqueueing to buffer.", info.rc)
            except Exception as e:
                logger.debug("Exception during publish: %s. Enqueueing to buffer.", e)

        # Disconnected or failed publish: Buffer message
        return self._enqueue_message(target_topic, payload_str)

    def _enqueue_message(self, topic: str, payload_str: str) -> bool:
        """Appends message to the thread-safe circular buffer."""
        with self._buffer_lock:
            # Check if buffer is already at max capacity
            if len(self._buffer) >= self.config.buffer_capacity:
                self._total_dropped += 1
                if self.config.overflow_policy == BufferOverflowPolicy.DROP_NEWEST:
                    # Drop new message
                    logger.warning("Telemetry buffer full (%d). Policy DROP_NEWEST: discarded incoming message.", self.config.buffer_capacity)
                    return False
                # Default DROP_OLDEST (deque with maxlen automatically drops oldest on append)

            self._buffer.append({"topic": topic, "payload": payload_str, "enqueued_at": time.time()})
            self._total_buffered += 1

        if self._state != ConnectionState.DEGRADED_BUFFERING:
            self._state = ConnectionState.DEGRADED_BUFFERING

        return True

    def _flush_buffer(self) -> int:
        """Flushes buffered messages to broker upon reconnection."""
        flushed_count = 0
        with self._buffer_lock:
            while self._buffer and not self._stop_event.is_set():
                item = self._buffer.popleft()
                try:
                    info = self._client.publish(item["topic"], item["payload"], qos=self.config.qos)
                    if info.rc == mqtt.MQTT_ERR_SUCCESS:
                        flushed_count += 1
                        self._total_published += 1
                    else:
                        # Reconnection interrupted; put item back at the front and break
                        self._buffer.appendleft(item)
                        break
                except Exception:
                    self._buffer.appendleft(item)
                    break

        if flushed_count > 0:
            logger.info("Flushed %d buffered telemetry messages to broker successfully.", flushed_count)
        return flushed_count

    def _check_outage_warning(self) -> None:
        """Checks if sustained disconnection >= 60 seconds and triggers explicit alert."""
        if self._state in (ConnectionState.DISCONNECTED, ConnectionState.DEGRADED_BUFFERING):
            duration = self.outage_duration_sec
            if duration >= self.config.failure_warning_threshold_sec and not self._outage_warning_triggered:
                self._outage_warning_triggered = True
                msg = (
                    f"CRITICAL TELEMETRY ALERT: Sustained MQTT disconnection for {duration:.1f}s "
                    f"(>= {self.config.failure_warning_threshold_sec}s threshold). "
                    f"Buffer holds {self.buffer_size}/{self.config.buffer_capacity} messages."
                )
                logger.error(msg)
                if self.on_outage_warning_callback:
                    try:
                        self.on_outage_warning_callback(duration)
                    except Exception as e:
                        logger.error("Error invoking outage warning callback: %s", e)

    def _reconnection_loop(self) -> None:
        """Background worker handling exponential backoff reconnection with jitter."""
        while not self._stop_event.is_set():
            if self._state in (ConnectionState.DISCONNECTED, ConnectionState.DEGRADED_BUFFERING):
                self._check_outage_warning()

                # Attempt reconnect
                success = self._trigger_connect()
                if success:
                    # Successful connect
                    self._current_backoff = self.config.backoff_initial_sec
                else:
                    # Apply exponential backoff with jitter
                    jitter = random.uniform(-self.config.backoff_jitter, self.config.backoff_jitter)
                    sleep_time = min(self.config.backoff_max_sec, self._current_backoff * (1.0 + jitter))
                    self._current_backoff = min(self.config.backoff_max_sec, self._current_backoff * self.config.backoff_factor)
                    time.sleep(max(0.5, sleep_time))
            else:
                time.sleep(1.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry health and buffer statistics."""
        return {
            "connection_state": self._state.value,
            "broker": f"{self.config.broker_host}:{self.config.broker_port}",
            "topic": self.config.topic,
            "buffer_capacity": self.config.buffer_capacity,
            "current_buffer_size": self.buffer_size,
            "total_published": self._total_published,
            "total_buffered": self._total_buffered,
            "total_dropped": self._total_dropped,
            "outage_duration_sec": round(self.outage_duration_sec, 2),
            "outage_warning_active": self._outage_warning_triggered,
        }
