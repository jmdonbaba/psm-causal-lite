"""pytest configuration for psm-causal-lite."""

import warnings

import pytest


@pytest.fixture(autouse=True)
def _suppress_sklearn_convergence():
    """Suppress sklearn ConvergenceWarning noise during tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        yield
