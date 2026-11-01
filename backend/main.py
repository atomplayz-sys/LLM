import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend import routes
from src.config import settings
from src.inference.generate import LLMGenerator

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Personal LLM API...")
    logger.info("Config: %s", settings.summary())
    routes.generator = LLMGenerator(settings.BASE_MODEL, settings.ADAPTER_PATH)
    if settings.PRELOAD_MODEL:
        logger.info("Pre-loading model...")
        routes.generator.load()
        logger.info("Model pre-loaded and ready!")
    else:
        logger.info("Model will lazy-load on first request")
    yield
    logger.info("Shutting down Personal LLM API...")
    routes.generator = None


app = FastAPI(
    title="Personal LLM API",
    description="A personal Large Language Model fine-tuned on the Open-Platypus dataset using LoRA. Built with TinyLlama-1.1B as the base model.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else "/redoc",
    openapi_url=None if settings.ENVIRONMENT == "production" else "/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
MAX_BODY_SIZE = 1048576


@app.middleware("http")
async def limit_body_size(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        from starlette.responses import JSONResponse

        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Maximum size is 1 MB."},
        )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
app.include_router(routes.router, prefix="/api/v1")
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/chat", response_class=HTMLResponse, tags=["UI"])
@app.get("/app", response_class=HTMLResponse, tags=["UI"])
async def chat_ui():
    index_file = static_dir / "index.html"
    return FileResponse(index_file)


@app.get("/", tags=["Root"])
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        index_file = static_dir / "index.html"
        return FileResponse(index_file)
    return {
        "message": "🧠 Personal LLM API",
        "version": "0.1.0",
        "docs": "/docs",
        "chat_ui": "/chat",
        "endpoints": {
            "generate": "/api/v1/generate",
            "chat": "/api/v1/chat",
            "health": "/api/v1/health",
            "model_info": "/api/v1/model/info",
        },
    }
