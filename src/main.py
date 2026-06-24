from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import chat, completions, health, models, tools
from src.config import ensure_data_dirs, get_settings
from src.inference.service import InferenceService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    ensure_data_dirs(settings)

    svc = InferenceService(settings)
    svc.initialize()

    # Wire the service into each router
    health.init_health(svc)
    models.init_models(svc)
    chat.init_chat(svc)
    completions.init_completions(svc)

    logger.info("startup_complete", extra={"model_id": settings.model_id})
    yield
    logger.info("shutdown")


app = FastAPI(
    title="llama-openai-server",
    description="Minimal OpenAI-compatible API backed by llama.cpp (ROCm/AMD GPU).",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc.detail), "code": exc.status_code}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"message": "Request validation failed", "details": exc.errors()}},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")
app.include_router(chat.router, prefix="/v1")
app.include_router(completions.router, prefix="/v1")
app.include_router(tools.router, prefix="/v1")
