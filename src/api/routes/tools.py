"""GET /v1/tools — Built-in tool discovery endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.tools import BUILTIN_REGISTRY

router = APIRouter(tags=["tools"])


@router.get("/tools")
def list_tools() -> dict[str, Any]:
    """
    Returns all built-in tools that are automatically available to the model
    on every chat completion request. Clients do not need to send these in the
    ``tools`` field — they are always active and the model infers when to call them.
    """
    schemas = BUILTIN_REGISTRY.schemas()
    return {
        "object": "list",
        "count": len(schemas),
        "always_active": True,
        "data": [
            {
                "type": s["type"],
                "function": s["function"],
                "source": "builtin",
            }
            for s in schemas
        ],
    }
