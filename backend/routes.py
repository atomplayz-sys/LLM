import asyncio
import hmac
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from backend.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.config import settings
from src.data.dataset_loader import get_dataset_info
from src.inference.generate import LLMGenerator

logger = logging.getLogger(__name__)
router = APIRouter()
generator: LLMGenerator = None


def get_generator() -> LLMGenerator:
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Server is still starting up.",
        )
    return generator


def verify_api_key(x_api_key: str | None = None):
    if not settings.is_auth_enabled():
        return
    if not hmac.compare_digest(x_api_key or "", settings.API_KEY):
        logger.warning("Rejected request with invalid API key")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    gen = get_generator()
    return HealthResponse(status="healthy", model_loaded=gen._loaded)


@router.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info(x_api_key: str | None = Header(None)):
    verify_api_key(x_api_key)
    gen = get_generator()
    info = gen.get_model_info()
    return ModelInfoResponse(
        base_model=info["base_model"],
        adapter_path=info["adapter_path"],
        loaded=info["loaded"],
        device=info["device"],
        parameters=info["parameters"],
        dataset=get_dataset_info(),
    )


@router.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(
    request: Request, body: GenerateRequest, x_api_key: str | None = Header(None)
):
    verify_api_key(x_api_key)
    gen = get_generator()
    try:
        response = await asyncio.to_thread(
            gen.generate,
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            repetition_penalty=body.repetition_penalty,
        )
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during generation. Please try again.",
        ) from None
    return GenerateResponse(
        prompt="[received]", response=response, model=gen.base_model_name
    )


@router.post("/chat", response_model=ChatResponse, tags=["Inference"])
async def chat_completion(
    request: Request, body: ChatRequest, x_api_key: str | None = Header(None)
):
    verify_api_key(x_api_key)
    gen = get_generator()
    messages = [{"role": msg.role, "content": msg.content} for msg in body.messages]
    try:
        response = await asyncio.to_thread(
            gen.chat,
            messages=messages,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
        )
    except Exception:
        logger.exception("Chat generation failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during chat generation. Please try again.",
        ) from None
    return ChatResponse(content=response, model=gen.base_model_name)
