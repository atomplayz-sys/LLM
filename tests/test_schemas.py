import pytest
from pydantic import ValidationError

from backend.schemas import (
    ChatMessage,
    ChatRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)


class TestGenerateRequest:
    def test_valid_request(self):
        req = GenerateRequest(prompt="Hello world")
        assert req.prompt == "Hello world"
        assert req.max_new_tokens == 512
        assert req.temperature == 0.7

    def test_custom_parameters(self):
        req = GenerateRequest(
            prompt="Test", max_new_tokens=100, temperature=0.5, top_p=0.8
        )
        assert req.max_new_tokens == 100
        assert req.temperature == 0.5

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_oversized_prompt_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="A" * 10001)

    def test_boundary_prompt_accepted(self):
        req = GenerateRequest(prompt="A" * 10000)
        assert len(req.prompt) == 10000

    def test_temperature_range(self):
        GenerateRequest(prompt="test", temperature=0.0)
        GenerateRequest(prompt="test", temperature=2.0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=2.1)

    def test_max_tokens_range(self):
        GenerateRequest(prompt="test", max_new_tokens=1)
        GenerateRequest(prompt="test", max_new_tokens=2048)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", max_new_tokens=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", max_new_tokens=2049)


class TestChatSchemas:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hi there")
        assert msg.role == "user"
        assert msg.content == "Hi there"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="system", content="test")

    def test_valid_chat_request(self):
        req = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
                ChatMessage(role="user", content="How are you?"),
            ]
        )
        assert len(req.messages) == 3

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_oversized_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="B" * 5001)

    def test_too_many_messages_rejected(self):
        messages = [ChatMessage(role="user", content="Hi")] * 51
        with pytest.raises(ValidationError):
            ChatRequest(messages=messages)


class TestResponseSchemas:
    def test_health_response(self):
        resp = HealthResponse(status="healthy", model_loaded=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_generate_response(self):
        resp = GenerateResponse(
            prompt="[received]", response="Hello!", model="TinyLlama"
        )
        assert resp.prompt == "[received]"
        assert resp.response == "Hello!"
