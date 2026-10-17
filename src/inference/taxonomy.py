"""Taxonomy Translation Engine for Team 5B.

Handles translation between domain-specific local class indices and the canonical
global 6-class taxonomy. Loaded directly from configs/taxonomy.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

# Canonical fallback taxonomy in case config file is unreachable
FALLBACK_GLOBAL_TAXONOMY: Dict[int, str] = {
    0: "healthy_crop",
    1: "dead_crop",
    2: "cargo_ship",
    3: "passenger_ship",
    4: "open",
    5: "short",
}

FALLBACK_DOMAIN_MAPPINGS = {
    "agriculture": {
        "local_classes": {0: "healthy_crop", 1: "dead_crop"},
        "local_to_global": {0: 0, 1: 1},
    },
    "ships": {
        "local_classes": {0: "cargo_ship", 1: "passenger_ship"},
        "local_to_global": {0: 2, 1: 3},
    },
    "space_electronics": {
        "local_classes": {0: "open", 1: "short"},
        "local_to_global": {0: 4, 1: 5},
    },
}


class TaxonomyManager:
    """Manages local-to-global class translations across domains."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # Default to repo configs/taxonomy.yaml
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "taxonomy.yaml"

        self.config_path = config_path
        self.global_taxonomy: Dict[int, str] = dict(FALLBACK_GLOBAL_TAXONOMY)
        self.domain_mappings = dict(FALLBACK_DOMAIN_MAPPINGS)
        self._load_config()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data and "global_taxonomy" in data:
                self.global_taxonomy = {int(k): str(v) for k, v in data["global_taxonomy"].items()}

            if data and "domain_mappings" in data:
                for domain, spec in data["domain_mappings"].items():
                    if "local_classes" in spec and "local_to_global" in spec:
                        self.domain_mappings[domain] = {
                            "local_classes": {int(k): str(v) for k, v in spec["local_classes"].items()},
                            "local_to_global": {int(k): int(v) for k, v in spec["local_to_global"].items()},
                        }
        except Exception as e:
            # Keep fallback if parse fails
            pass

    def to_global(self, domain: str, local_class_id: int) -> Tuple[int, str]:
        """Translates a domain local class ID into (global_class_id, global_class_name)."""
        domain_key = domain.lower()
        if domain_key not in self.domain_mappings:
            raise ValueError(f"Unknown domain '{domain}'. Supported: {list(self.domain_mappings.keys())}")

        mapping = self.domain_mappings[domain_key]["local_to_global"]
        if local_class_id not in mapping:
            raise ValueError(
                f"Unknown local class ID {local_class_id} for domain '{domain}'. "
                f"Valid local IDs: {list(mapping.keys())}"
            )

        global_id = mapping[local_class_id]
        global_name = self.global_taxonomy.get(global_id, f"unknown_class_{global_id}")
        return global_id, global_name

    def get_local_classes(self, domain: str) -> Dict[int, str]:
        """Retrieves all local class mappings {local_id: name} for a given domain."""
        domain_key = domain.lower()
        if domain_key not in self.domain_mappings:
            raise ValueError(f"Unknown domain '{domain}'. Supported: {list(self.domain_mappings.keys())}")
        return dict(self.domain_mappings[domain_key]["local_classes"])

    def get_local_name(self, domain: str, local_class_id: int) -> str:
        """Retrieves the domain-internal class label for a local class ID."""
        domain_key = domain.lower()
        if domain_key not in self.domain_mappings:
            raise ValueError(f"Unknown domain '{domain}'.")

        local_classes = self.domain_mappings[domain_key]["local_classes"]
        return local_classes.get(local_class_id, f"local_class_{local_class_id}")


# Global singleton instance
taxonomy_manager = TaxonomyManager()
