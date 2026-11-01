"""
Pydantic Schemas — Request/Response models for the API.
"""

from pydantic import BaseModel, Field

# ─── Request Schemas ──────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(
        ...,
        description="The input prompt to generate a response for",
        min_length=1,
        max_length=10000,
        examples=["Explain quantum computing in simple terms"],
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 = deterministic, higher = more creative)",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling",
    )
    repetition_penalty: float = Field(
        default=1.15,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens",
    )


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(
        ...,
        description="Role of the message sender",
        pattern="^(user|assistant)$",
    )
    content: str = Field(
        ...,
        description="The message content",
        min_length=1,
        max_length=5000,
    )


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    messages: list[ChatMessage] = Field(
        ...,
        description="List of chat messages in conversation order",
        min_length=1,
        max_length=50,
    )
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)


# ─── Response Schemas ─────────────────────────────────────────────


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""

    prompt: str
    response: str
    model: str
    tokens_generated: int | None = None


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    role: str = "assistant"
    content: str
    model: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response body for the /model/info endpoint."""

    base_model: str
    adapter_path: str
    loaded: bool
    device: str
    parameters: int | None = None
    dataset: dict
