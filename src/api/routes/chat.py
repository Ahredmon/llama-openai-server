"""POST /v1/chat/completions — OpenAI Chat Completions API."""
from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Any, Generator, Literal, Union

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.inference.service import GenerationControls, InferenceService

router = APIRouter(tags=["chat"])

_svc: InferenceService | None = None


def init_chat(svc: InferenceService) -> None:
    global _svc
    _svc = svc


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str = Field(min_length=1)


class ImageUrlContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[
    Union[TextContentPart, ImageUrlContentPart],
    Field(discriminator="type"),
]

MessageContent = Union[str, list[ContentPart]]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: MessageContent

    @field_validator("content")
    @classmethod
    def _not_empty(cls, v: MessageContent) -> MessageContent:
        if isinstance(v, str) and not v:
            raise ValueError("content must not be empty")
        if isinstance(v, list) and not v:
            raise ValueError("content parts list must not be empty")
        return v


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: Union[list[str], str, None] = None
    seed: int | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)

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


class OAIMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _content_to_str(content: MessageContent) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, TextContentPart):
            parts.append(part.text)
        elif isinstance(part, ImageUrlContentPart):
            parts.append("[image]")
    return "\n".join(parts)


def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    result = []
    for m in messages:
        if isinstance(m.content, str):
            content: Any = m.content
        else:
            has_images = any(isinstance(p, ImageUrlContentPart) for p in m.content)
            if has_images:
                content = [
                    {"type": "text", "text": p.text}
                    if isinstance(p, TextContentPart)
                    else {"type": "image_url", "image_url": {"url": p.image_url.url}}
                    for p in m.content
                ]
            else:
                content = _content_to_str(m.content)
        result.append({"role": m.role, "content": content})
    return result


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_stream(
    raw: Generator[dict, None, None],
    model: str,
    completion_id: str,
) -> Generator[str, None, None]:
    created = int(time.time())
    for chunk in raw:
        delta = chunk.get("delta", "")
        finish_reason = chunk.get("finish_reason")
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": (
                        {"role": "assistant", "content": delta} if delta else {}
                    ),
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    assert _svc is not None, "InferenceService not initialized"
    model_id = _svc.settings.model_id
    controls = request.to_controls()
    messages = _messages_to_dicts(request.messages)

    if request.stream:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        raw = _svc.stream_chat_complete(messages, controls)
        return StreamingResponse(
            _sse_stream(raw, model_id, cid),
            media_type="text/event-stream",
        )

    result = _svc.chat_complete(messages, controls)
    return ChatCompletionResponse(
        id=result["id"],
        created=result["created"],
        model=model_id,
        choices=[
            OAIChoice(
                message=OAIMessage(content=result["output"]),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=OAIUsage(**result["usage"]),
    )
"""POST /v1/chat/completions — OpenAI Chat Completions API."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Generator, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.inference.service import GenerationControls, InferenceService

router = APIRouter(tags=["chat"])

_svc: InferenceService | None = None


def init_chat(svc: InferenceService) -> None:
    global _svc
    _svc = svc


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class ChatMessageRequest(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessageRequest] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    seed: int | None = None
    stream: bool = False
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)

    def _stop_list(self) -> list[str]:
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop or []

    def to_controls(self) -> GenerationControls:
        defaults = GenerationControls()
        return GenerationControls(
            temperature=self.temperature if self.temperature is not None else defaults.temperature,
            top_p=self.top_p if self.top_p is not None else defaults.top_p,
            max_tokens=self.max_tokens if self.max_tokens is not None else defaults.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self._stop_list(),
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


def _sse_stream(
    raw: Generator[dict[str, Any], None, None],
    model: str,
    completion_id: str,
) -> Generator[str, None, None]:
    created = int(time.time())
    for chunk in raw:
        delta = chunk.get("delta", "")
        finish_reason = chunk.get("finish_reason")
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta} if delta else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    assert _svc is not None, "Service not initialized"
    messages = [m.model_dump() for m in request.messages]
    controls = request.to_controls()
    model = _svc.settings.model_id

    if request.stream:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        raw = _svc.stream_chat_complete(messages, controls)
        return StreamingResponse(
            _sse_stream(raw, model, completion_id),
            media_type="text/event-stream",
        )

    result = _svc.chat_complete(messages, controls)
    return ChatCompletionResponse(
        id=result["id"],
        created=result["created"],
        model=result["model"],
        choices=[
            Choice(
                message=ChatMessage(content=result["output"]),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=UsageInfo(**result["usage"]),
    )
