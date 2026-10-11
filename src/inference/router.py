"""Unified Inference Router for Team 5B.

Implements the central orchestrator that dispatches inference requests to domain-specific
adapters (Agriculture, Ships, Space Electronics), normalizes outputs, and produces
telemetry-ready payloads without cross-domain prediction pollution.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union
from .adapters import (
    ADAPTER_REGISTRY,
    AgricultureAdapter,
    BaseModelAdapter,
    ShipsAdapter,
    SpaceElectronicsAdapter,
)
from .schema import Domain, InferenceRequest, InferenceResult

logger = logging.getLogger("InferenceRouter")


class UnifiedInferenceRouter:
    """Unified Orchestrator across domain-specific vision models."""

    def __init__(self, lazy_load: bool = True, device: str = "cpu"):
        self.device = device
        self.lazy_load = lazy_load
        self._adapters: Dict[str, BaseModelAdapter] = {}
        self._initialize_adapters()

    def _initialize_adapters(self) -> None:
        """Instantiates domain adapters."""
        self._adapters = {
            "agriculture": AgricultureAdapter(device=self.device),
            "ships": ShipsAdapter(device=self.device),
            "space_electronics": SpaceElectronicsAdapter(device=self.device),
        }
        if not self.lazy_load:
            for domain, adapter in self._adapters.items():
                logger.info("Pre-loading adapter for domain: %s", domain)
                adapter.load()

    def get_adapter(self, domain: Union[str, Domain]) -> BaseModelAdapter:
        """Retrieves adapter for a specific domain."""
        domain_key = domain.value if isinstance(domain, Domain) else str(domain).lower()
        if domain_key not in self._adapters:
            raise ValueError(
                f"Unsupported domain '{domain_key}'. "
                f"Valid domains: {list(self._adapters.keys())}"
            )
        return self._adapters[domain_key]

    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Routes inference request to the designated domain adapter.

        Args:
            request: Standardized InferenceRequest with domain specified.

        Returns:
            InferenceResult with canonical taxonomy and metadata.
        """
        if request.domain is None:
            raise ValueError(
                "InferenceRequest must specify a 'domain' (e.g. 'agriculture', 'ships', 'space_electronics'). "
                "Direct cross-domain weight merging is rejected per ADR-001."
            )

        adapter = self.get_adapter(request.domain)
        logger.debug("Routing request [frame_id=%s] to adapter: %s", request.frame_id, adapter.model_name)
        return adapter.predict(request)

    def run_all_domains(self, request: InferenceRequest) -> Dict[str, InferenceResult]:
        """Runs image through all three domain models independently for comparative research.

        Note: Results are returned as separate domain entries and NOT merged post-NMS,
        avoiding cross-domain false positives per ADR-001.
        """
        results = {}
        for domain_name, adapter in self._adapters.items():
            req_copy = request.model_copy()
            req_copy.domain = Domain(domain_name)
            results[domain_name] = adapter.predict(req_copy)
        return results

    def predict_telemetry(self, request: InferenceRequest) -> dict:
        """Executes inference and returns telemetry-ready JSON payload for MQTT streaming."""
        result = self.predict(request)
        return result.to_telemetry_payload()
