from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "0.0.0.0"
    port: int = 8000

    # Model identity and path resolution
    model_id: str = "local-model"
    model_path: str | None = None
    models_dir: str = "models"
    # HuggingFace auto-download (optional)
    hf_repo_id: str | None = None
    hf_model_filename: str | None = None
    hf_revision: str = "main"

    # Multimodal / vision — CLIP mmproj for Gemma 4 and compatible vision models
    clip_model_path: str | None = None
    hf_clip_repo_id: str | None = None
    hf_clip_filename: str | None = None

    # llama.cpp backend parameters
    n_gpu_layers: int = 80
    n_ctx: int = 0  # 0 = auto-detect from VRAM
    n_threads: int = 8
    n_threads_batch: int | None = None
    n_batch: int = 1024
    n_ubatch: int = 512

    # AMD/ROCm memory and performance optimizations
    mlock: bool = True
    offload_kqv: bool = True
    flash_attn: bool = False
    f16_kv: bool = True

    # Defaults used when a request omits generation parameters
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_repeat_penalty: float = 1.1
    default_max_tokens: int = 256

    log_level: str = "INFO"

    # Set False to disable ROCm check (e.g. for testing)
    rocm_required: bool = True
    allow_mock_backend: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def ensure_data_dirs(settings: Settings) -> None:
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
