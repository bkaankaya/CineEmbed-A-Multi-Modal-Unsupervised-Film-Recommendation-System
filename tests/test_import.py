"""Sanity test — verifies the package installs and exposes its version."""
import cineembed


def test_import_and_version():
    assert hasattr(cineembed, '__version__')
    assert cineembed.__version__ == "0.1.0"
