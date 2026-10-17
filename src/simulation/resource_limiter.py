"""Edge Resource Constraint Simulator for Team 5B.

Conforms to Requirement REQ-08 and Milestone B8:
- Simulates edge-device computational constraints (CPU throttle duty cycle).
- Simulates network transmission latency (e.g. cellular / satellite link delays).
- Enforces and monitors memory budget caps (RSS RAM ceilings).
- Enables reproducible benchmarking under edge hardware resource profiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple
import psutil

logger = logging.getLogger("ResourceSimulator")


@dataclass
class SimulationProfile:
    """Hardware profile specification of a simulated edge deployment target."""

    name: str = "Generic-Edge-Unit"
    cpu_throttle_percent: float = 0.0  # 0.0 = full speed, 50.0 = 50% throttled CPU
    simulated_network_delay_ms: float = 0.0  # Added latency in milliseconds
    memory_limit_mb: float = 1024.0  # RAM budget ceiling in MB

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeResourceSimulator:
    """Manages simulated constraints during model inference and telemetry dispatch."""

    def __init__(self, profile: Optional[SimulationProfile] = None):
        self.profile = profile or SimulationProfile()
        self.process = psutil.Process()

    def check_memory_budget(self) -> Tuple[bool, float, float]:
        """Checks if current process memory exceeds the simulated edge budget."""
        current_rss = self.process.memory_info().rss / (1024 * 1024)
        within_budget = current_rss <= self.profile.memory_limit_mb
        return within_budget, round(current_rss, 2), self.profile.memory_limit_mb

    def apply_cpu_throttle(self, work_duration_sec: float) -> float:
        """Injects proportional sleep to simulate throttled CPU compute capacity."""
        if self.profile.cpu_throttle_percent <= 0.0:
            return 0.0

        # If 50% throttled, sleep duration equals work duration
        # throttle_ratio = throttle% / (100 - throttle%)
        throttle_pct = min(90.0, max(0.0, self.profile.cpu_throttle_percent))
        delay = work_duration_sec * (throttle_pct / (100.0 - throttle_pct))
        if delay > 0:
            time.sleep(delay)
        return delay

    def apply_network_delay(self) -> float:
        """Injects simulated network latency in seconds."""
        if self.profile.simulated_network_delay_ms <= 0.0:
            return 0.0
        delay_sec = self.profile.simulated_network_delay_ms / 1000.0
        time.sleep(delay_sec)
        return delay_sec
