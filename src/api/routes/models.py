from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.inference.service import InferenceService

router = APIRouter(tags=["models"])

_svc: InferenceService | None = None


def init_models(svc: InferenceService) -> None:
    global _svc
    _svc = svc


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    vision: bool = False


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


@router.get("/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    if _svc is None:
        return ModelsListResponse(data=[])
    meta = _svc.model_metadata()
    backend = getattr(_svc, "_backend", None)
    vision = getattr(backend, "vision_enabled", False)
    return ModelsListResponse(
        data=[
            ModelObject(
                id=meta["id"],
                created=int(time.time()),
                owned_by=meta.get("owned_by", "local"),
                vision=vision,
            )
        ]
    )
