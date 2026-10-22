"""Smoke test — verifies the project skeleton imports correctly."""

import config


def test_config_imports():
    """config module loads without error and exposes expected attributes."""
    assert hasattr(config, "SEMANTIC_SCHOLAR_API_KEY")
    assert hasattr(config, "OPENALEX_EMAIL")
    assert hasattr(config, "OPENALEX_API_KEY")
    assert hasattr(config, "UNPAYWALL_EMAIL")
