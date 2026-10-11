"""Unit Tests for UnifiedInferenceRouter."""

import pytest
from src.inference.router import UnifiedInferenceRouter
from src.inference.schema import Domain, InferenceRequest


def test_router_initialization():
    router = UnifiedInferenceRouter(lazy_load=True)
    assert "agriculture" in router._adapters
    assert "ships" in router._adapters
    assert "space_electronics" in router._adapters

    ag_adapter = router.get_adapter(Domain.AGRICULTURE)
    assert ag_adapter.domain == "agriculture"
    assert ag_adapter.target_imgsz == 512

    ships_adapter = router.get_adapter("ships")
    assert ships_adapter.domain == "ships"
    assert ships_adapter.target_imgsz == 640

    elec_adapter = router.get_adapter(Domain.SPACE_ELECTRONICS)
    assert elec_adapter.domain == "space_electronics"
    assert elec_adapter.target_imgsz == 640


def test_router_invalid_domain():
    router = UnifiedInferenceRouter(lazy_load=True)
    with pytest.raises(ValueError, match="Unsupported domain"):
        router.get_adapter("robotics")


def test_router_missing_domain_in_request():
    router = UnifiedInferenceRouter(lazy_load=True)
    req = InferenceRequest(image="dummy_path.jpg", domain=None)
    with pytest.raises(ValueError, match="must specify a 'domain'"):
        router.predict(req)
