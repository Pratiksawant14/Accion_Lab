"""Unit Tests for TaxonomyManager and Class Mappings."""

import pytest
from src.inference.taxonomy import taxonomy_manager


def test_taxonomy_global_mappings():
    # Agriculture mappings
    g_id, g_name = taxonomy_manager.to_global("agriculture", 0)
    assert g_id == 0
    assert g_name == "healthy_crop"

    g_id, g_name = taxonomy_manager.to_global("agriculture", 1)
    assert g_id == 1
    assert g_name == "dead_crop"

    # Ships mappings
    g_id, g_name = taxonomy_manager.to_global("ships", 0)
    assert g_id == 2
    assert g_name == "cargo_ship"

    g_id, g_name = taxonomy_manager.to_global("ships", 1)
    assert g_id == 3
    assert g_name == "passenger_ship"

    # Space Electronics mappings
    g_id, g_name = taxonomy_manager.to_global("space_electronics", 0)
    assert g_id == 4
    assert g_name == "open"

    g_id, g_name = taxonomy_manager.to_global("space_electronics", 1)
    assert g_id == 5
    assert g_name == "short"


def test_taxonomy_invalid_domain():
    with pytest.raises(ValueError, match="Unknown domain"):
        taxonomy_manager.to_global("nonexistent_domain", 0)


def test_taxonomy_invalid_local_class_id():
    with pytest.raises(ValueError, match="Unknown local class ID"):
        taxonomy_manager.to_global("agriculture", 999)


def test_taxonomy_local_names():
    assert taxonomy_manager.get_local_name("agriculture", 0) == "healthy_crop"
    assert taxonomy_manager.get_local_name("ships", 1) == "passenger_ship"
    assert taxonomy_manager.get_local_name("space_electronics", 0) == "open"
