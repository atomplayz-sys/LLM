"""
Tests for Pydantic request/response schemas.

Tests cover:
  - Valid request construction
  - Default values
  - Input validation (min/max length, value ranges)
  - Edge cases and boundary values
"""

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
    """Tests for the GenerateRequest schema."""

    def test_valid_request(self):
        req = GenerateRequest(prompt="Hello world")
        assert req.prompt == "Hello world"
        assert req.max_new_tokens == 512
        assert req.temperature == 0.7

    def test_custom_parameters(self):
        req = GenerateRequest(
            prompt="Test",
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
        )
        assert req.max_new_tokens == 100
        assert req.temperature == 0.5

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_oversized_prompt_rejected(self):
        """Prompts exceeding max_length=10000 should be rejected."""
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="A" * 10001)

    def test_boundary_prompt_accepted(self):
        """Exactly max_length=10000 characters should be accepted."""
        req = GenerateRequest(prompt="A" * 10000)
        assert len(req.prompt) == 10000

    def test_temperature_range(self):
        """Temperature must be between 0.0 and 2.0."""
        GenerateRequest(prompt="test", temperature=0.0)  # min
        GenerateRequest(prompt="test", temperature=2.0)  # max

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=-0.1)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=2.1)

    def test_max_tokens_range(self):
        """max_new_tokens must be between 1 and 2048."""
        GenerateRequest(prompt="test", max_new_tokens=1)  # min
        GenerateRequest(prompt="test", max_new_tokens=2048)  # max

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", max_new_tokens=0)
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", max_new_tokens=2049)


class TestChatSchemas:
    """Tests for ChatMessage and ChatRequest schemas."""

    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hi there")
        assert msg.role == "user"
        assert msg.content == "Hi there"

    def test_invalid_role_rejected(self):
        """Only 'user' and 'assistant' roles are allowed."""
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
        """Message content exceeding max_length=5000 should be rejected."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="B" * 5001)

    def test_too_many_messages_rejected(self):
        """More than 50 messages should be rejected."""
        messages = [ChatMessage(role="user", content="Hi")] * 51
        with pytest.raises(ValidationError):
            ChatRequest(messages=messages)


class TestResponseSchemas:
    """Tests for response schemas."""

    def test_health_response(self):
        resp = HealthResponse(status="healthy", model_loaded=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_generate_response(self):
        resp = GenerateResponse(
            prompt="[received]",
            response="Hello!",
            model="TinyLlama",
        )
        assert resp.prompt == "[received]"
        assert resp.response == "Hello!"
