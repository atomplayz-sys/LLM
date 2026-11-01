import pytest


@pytest.fixture
def sample_instruction():
    return {"instruction": "What is 2+2?", "input": "", "output": "4"}


@pytest.fixture
def sample_instruction_with_context():
    return {
        "instruction": "Summarize the text",
        "input": "Some long text here",
        "output": "Summary",
    }


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from backend import routes
    from backend.main import app
    from src.inference.generate import LLMGenerator

    routes.generator = LLMGenerator()
    with TestClient(app) as client:
        yield client
    routes.generator = None
