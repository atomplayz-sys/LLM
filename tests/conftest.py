"""
Shared test fixtures for the Personal LLM test suite.

Fixtures defined here are automatically available to all test files
in the tests/ directory without needing explicit imports.
"""

import pytest


@pytest.fixture
def sample_instruction():
    """A standard Open-Platypus-style sample for testing."""
    return {
        "instruction": "What is 2+2?",
        "input": "",
        "output": "4",
    }


@pytest.fixture
def sample_instruction_with_context():
    """A sample with the optional 'input' context field."""
    return {
        "instruction": "Summarize the text",
        "input": "Some long text here",
        "output": "Summary",
    }


@pytest.fixture
def api_client():
    """
    FastAPI TestClient with a pre-initialized (unloaded) generator.

    The generator is created but the model is NOT loaded into memory,
    so endpoints that check `generator is not None` will pass, but
    actual inference calls will trigger lazy loading.
    """
    from fastapi.testclient import TestClient

    from backend import routes
    from backend.main import app
    from src.inference.generate import LLMGenerator

    # Set up a generator so endpoints don't return 503
    routes.generator = LLMGenerator()

    with TestClient(app) as client:
        yield client

    # Teardown
    routes.generator = None
