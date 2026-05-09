from __future__ import annotations

from fastapi import APIRouter

from src.inference.service import InferenceService

router = APIRouter(tags=["health"])

_svc: InferenceService | None = None


def init_health(svc: InferenceService) -> None:
    global _svc
    _svc = svc


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def readiness() -> dict:
    ready = _svc.ready if _svc else False
    return {"status": "ready" if ready else "not_ready"}
