import os
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
load_dotenv(_env_path, override=False)


class Settings:
    BASE_MODEL: str = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ADAPTER_PATH: str = os.getenv("ADAPTER_PATH", "./output/final")
    PRELOAD_MODEL: bool = os.getenv("PRELOAD_MODEL", "false").lower() == "true"
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://localhost:5173",
    ).split(",")
    API_KEY: str | None = os.getenv("API_KEY")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    WANDB_API_KEY: str | None = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "personal-llm")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
    LOGGING_DIR: str = os.getenv("LOGGING_DIR", "./logs")
    CONFIG_PATH: str = os.getenv("CONFIG_PATH", "config/training_config.yaml")

    def is_auth_enabled(self) -> bool:
        return self.API_KEY is not None and len(self.API_KEY) > 0

    def summary(self) -> dict:
        return {
            "base_model": self.BASE_MODEL,
            "adapter_path": self.ADAPTER_PATH,
            "preload_model": self.PRELOAD_MODEL,
            "allowed_origins": self.ALLOWED_ORIGINS,
            "auth_enabled": self.is_auth_enabled(),
            "rate_limit": self.RATE_LIMIT,
            "hf_token_set": self.HF_TOKEN is not None,
            "wandb_configured": self.WANDB_API_KEY is not None,
            "output_dir": self.OUTPUT_DIR,
        }


settings = Settings()
