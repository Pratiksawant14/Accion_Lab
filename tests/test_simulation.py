"""Unit tests for Edge Resource Constraint Simulator (Milestone B8 / REQ-08)."""

import time
import pytest

from src.simulation.resource_limiter import EdgeResourceSimulator, SimulationProfile


def test_memory_budget_check():
    # Set huge memory limit (100 GB) -> within budget
    sim_large = EdgeResourceSimulator(SimulationProfile(memory_limit_mb=102400.0))
    within_budget, current_rss, limit = sim_large.check_memory_budget()
    assert within_budget is True
    assert current_rss > 0.0

    # Set tiny memory limit (1 MB) -> budget exceeded
    sim_tiny = EdgeResourceSimulator(SimulationProfile(memory_limit_mb=1.0))
    within_budget, current_rss, limit = sim_tiny.check_memory_budget()
    assert within_budget is False


def test_cpu_throttle_delay():
    # 50% throttled CPU: 0.05s work should inject ~0.05s sleep
    sim = EdgeResourceSimulator(SimulationProfile(cpu_throttle_percent=50.0))
    t0 = time.perf_counter()
    delay = sim.apply_cpu_throttle(work_duration_sec=0.04)
    t1 = time.perf_counter()

    assert delay > 0.03
    assert (t1 - t0) >= 0.03


def test_network_delay():
    sim = EdgeResourceSimulator(SimulationProfile(simulated_network_delay_ms=30.0))
    t0 = time.perf_counter()
    delay_sec = sim.apply_network_delay()
    t1 = time.perf_counter()

    assert delay_sec == 0.03
    assert (t1 - t0) >= 0.028
