"""POST /v1/completions — OpenAI Text Completions API."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Generator, Literal, Union

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.inference.service import GenerationControls, InferenceService

router = APIRouter(tags=["completions"])

_svc: InferenceService | None = None


def init_completions(svc: InferenceService) -> None:
    global _svc
    _svc = svc


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: Union[list[str], str, None] = None
    seed: int | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    # Optional system prompt injected before the user prompt
    system_prompt: str | None = None

    def to_controls(self) -> GenerationControls:
        defaults = GenerationControls()
        stop_list: list[str] = (
            [self.stop] if isinstance(self.stop, str)
            else (self.stop or [])
        )
        return GenerationControls(
            temperature=self.temperature if self.temperature is not None else defaults.temperature,
            top_p=self.top_p if self.top_p is not None else defaults.top_p,
            max_tokens=self.max_tokens if self.max_tokens is not None else defaults.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=stop_list,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OAICompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[OAICompletionChoice]
    usage: OAIUsage


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse_stream(
    raw: Generator[dict, None, None],
    model: str,
    completion_id: str,
) -> Generator[str, None, None]:
    created = int(time.time())
    for chunk in raw:
        finish_reason = chunk.get("finish_reason")
        payload = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": chunk.get("delta", ""),
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/completions")
def completions(request: CompletionRequest) -> Any:
    assert _svc is not None, "InferenceService not initialized"
    model_id = _svc.settings.model_id
    controls = request.to_controls()
    json_mode = request.response_format.type == "json_object"

    if request.stream:
        cid = f"cmpl-{uuid.uuid4().hex[:12]}"
        raw = _svc.stream_complete(
            request.prompt, controls, system_prompt=request.system_prompt, json_mode=json_mode
        )
        return StreamingResponse(
            _sse_stream(raw, model_id, cid),
            media_type="text/event-stream",
        )

    result = _svc.complete(
        request.prompt, controls, system_prompt=request.system_prompt, json_mode=json_mode
    )
    return CompletionResponse(
        id=result["id"],
        created=result["created"],
        model=model_id,
        choices=[
            OAICompletionChoice(
                text=result["output"],
                finish_reason=result["finish_reason"],
            )
        ],
        usage=OAIUsage(**result["usage"]),
    )
