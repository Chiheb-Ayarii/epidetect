"""Smoke test: the package is importable (keeps CI green before the real modules exist)."""

import epidetect


def test_package_has_version() -> None:
    assert isinstance(epidetect.__version__, str)
