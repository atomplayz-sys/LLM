"""
Tests for configuration and constants modules.

Tests cover:
  - Settings defaults
  - Constants integrity
  - Auth toggle logic
"""

from src.config import Settings
from src.constants import CONTROL_TOKENS, MAX_GENERATION_TOKENS, SYSTEM_PROMPT


class TestConstants:
    """Tests for shared constants."""

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_control_tokens_complete(self):
        """All required ChatML tokens should be defined."""
        assert "<|system|>" in CONTROL_TOKENS
        assert "<|user|>" in CONTROL_TOKENS
        assert "<|assistant|>" in CONTROL_TOKENS
        assert "</s>" in CONTROL_TOKENS

    def test_max_generation_tokens_reasonable(self):
        """Token cap should be between 128 and 4096."""
        assert 128 <= MAX_GENERATION_TOKENS <= 4096


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_base_model(self):
        s = Settings()
        assert "TinyLlama" in s.BASE_MODEL

    def test_default_environment(self):
        s = Settings()
        assert s.ENVIRONMENT == "development"

    def test_auth_disabled_by_default(self):
        """Without API_KEY set, authentication should be disabled."""
        s = Settings()
        # API_KEY defaults to None from env
        if s.API_KEY is None:
            assert not s.is_auth_enabled()

    def test_summary_does_not_leak_secrets(self):
        """The summary dict should never contain actual secret values."""
        s = Settings()
        summary = s.summary()
        # Should contain boolean flags, not actual values
        assert "hf_token_set" in summary
        assert "wandb_configured" in summary
        # Should NOT contain the actual tokens
        assert "HF_TOKEN" not in str(summary.values())
