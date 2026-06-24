"""POST /v1/chat/completions — OpenAI Chat Completions API."""
from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Any, Generator, Literal, Union

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from src.inference.service import GenerationControls, InferenceService
from src.tools import BUILTIN_REGISTRY

router = APIRouter(tags=["chat"])

# Max tool-call / tool-result iterations before returning whatever we have
_MAX_TOOL_ITERATIONS = 8

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
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> "ChatMessage":
        v = self.content
        if v is None:
            return self
        # Empty string content is valid for assistant messages that only carry
        # tool_calls, and for tool messages — the OpenAI spec permits it.
        if isinstance(v, str) and not v:
            if self.role not in ("assistant", "tool"):
                raise ValueError("content must not be empty")
            self.content = None
        elif isinstance(v, list) and not v:
            raise ValueError("content parts list must not be empty")
        return self


class JsonSchemaSpec(BaseModel):
    name: str
    schema: dict[str, Any] | None = None
    strict: bool | None = None
    description: str | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaSpec | None = None


class ToolFunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSpec


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


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
    tools: list[Tool] | None = None
    tool_choice: Union[Literal["none", "auto", "required"], dict[str, Any], None] = None
    parallel_tool_calls: bool | None = None

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
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


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
    system_tools: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def _content_to_str(content: MessageContent) -> str:
    """Flatten content to plain text (used only for tool-result messages)."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, TextContentPart):
            parts.append(part.text)
        elif isinstance(part, ImageUrlContentPart):
            parts.append("[image]")
    return "\n".join(parts)


def _content_for_llm(content: MessageContent | None) -> Any:
    """
    Convert message content to the format expected by llama.cpp.

    Plain-text and tool messages stay as a string.  Messages that contain
    at least one image part are passed as a list of dicts so the vision
    pipeline receives the actual base-64 payload rather than a placeholder.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    # If every part is plain text, keep it as a string for maximum compatibility
    # with non-vision models.
    has_image = any(isinstance(p, ImageUrlContentPart) for p in content)
    if not has_image:
        return _content_to_str(content)
    # Multimodal: pass the full part list as dicts
    return [
        (
            {"type": "text", "text": p.text}
            if isinstance(p, TextContentPart)
            else {"type": "image_url", "image_url": {"url": p.image_url.url}}
        )
        for p in content
    ]


def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {
            "role": m.role,
            "content": _content_for_llm(m.content),
        }
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls is not None:
            d["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_stream(
    raw: Generator[dict, None, None] | list[dict],
    model: str,
    completion_id: str,
    append_done: bool = True,
) -> Generator[str, None, None]:
    created = int(time.time())
    for chunk in raw:
        text_delta = chunk.get("delta", "")
        tool_call_chunks = chunk.get("tool_call_chunks")
        finish_reason = chunk.get("finish_reason")
        if tool_call_chunks is not None:
            delta_obj: dict[str, Any] = {"tool_calls": tool_call_chunks}
        else:
            delta_obj = (
                {"role": "assistant", "content": text_delta} if text_delta else {}
            )
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta_obj,
                    "finish_reason": finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    if append_done:
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Agentic tool-call loop helpers
# ---------------------------------------------------------------------------


def _all_tools(client_tools: list[dict] | None) -> list[dict]:
    """Merge server built-in schemas with any client-provided tool schemas."""
    return [*BUILTIN_REGISTRY.schemas(), *(client_tools or [])]


def _effective_tool_choice(client_choice: Any, client_tools: list[dict] | None) -> Any:
    """
    Built-in tools are always active with 'auto' intent — the model infers when
    to call them. A client-supplied 'none' only suppresses *their* explicit tools;
    it cannot disable server-side built-ins.
    """
    if client_choice == "none" and not client_tools:
        # Client said 'none' but has no tools of their own — keep auto for built-ins
        return "auto"
    return client_choice if client_choice is not None else "auto"


def _has_non_builtin(tool_calls: list[dict]) -> bool:
    return any(not BUILTIN_REGISTRY.has(tc["function"]["name"]) for tc in tool_calls)


def _execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute each tool call and return the corresponding tool-role messages."""
    messages: list[dict] = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = tc["function"].get("arguments", "")
        try:
            output = BUILTIN_REGISTRY.execute(name, args)
        except Exception as exc:  # noqa: BLE001
            output = json.dumps({"error": str(exc)})
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": output,
        })
    return messages


def _run_agentic_chat(
    messages: list[dict],
    controls: GenerationControls,
    client_tools: list[dict] | None,
    tool_choice: Any,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Non-streaming agentic loop: executes built-in tools until a final answer."""
    assert _svc is not None
    all_t = _all_tools(client_tools) or None
    current = list(messages)
    effective_choice: Any = _effective_tool_choice(tool_choice, client_tools)

    for _ in range(_MAX_TOOL_ITERATIONS):
        result = _svc.chat_complete(current, controls, tools=all_t, tool_choice=effective_choice, json_mode=json_mode)
        effective_choice = "auto"  # subsequent turns always auto

        tool_calls = result.get("tool_calls")
        if result["finish_reason"] != "tool_calls" or not tool_calls:
            return result
        if _has_non_builtin(tool_calls):
            # Contains client-managed tools — return to caller
            return result

        current.append({
            "role": "assistant",
            "content": result.get("output"),
            "tool_calls": tool_calls,
        })
        current.extend(_execute_tool_calls(tool_calls))

    return result


def _collect_stream_tool_calls(
    chunks: list[dict],
) -> list[dict] | None:
    """Reassemble tool calls from a collected list of raw stream chunks."""
    tc_accum: dict[int, dict] = {}
    for chunk in chunks:
        for tc_chunk in chunk.get("tool_call_chunks") or []:
            idx: int = tc_chunk.get("index", 0)
            if idx not in tc_accum:
                tc_accum[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            if tc_chunk.get("id"):
                tc_accum[idx]["id"] = tc_chunk["id"]
            fn = tc_chunk.get("function") or {}
            if fn.get("name"):
                tc_accum[idx]["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                tc_accum[idx]["function"]["arguments"] += fn["arguments"]
    return list(tc_accum.values()) if tc_accum else None


def _agentic_stream(
    messages: list[dict],
    controls: GenerationControls,
    client_tools: list[dict] | None,
    tool_choice: Any,
    model_id: str,
    cid: str,
    json_mode: bool = False,
) -> Generator[str, None, None]:
    """
    Streaming agentic loop. Streams tokens in real-time unless a tool call turn is active,
    in which case the tool call is buffered, executed silently (if built-in), or sent to client.
    """
    assert _svc is not None
    all_t = _all_tools(client_tools) or None
    current = list(messages)
    effective_choice: Any = _effective_tool_choice(tool_choice, client_tools)

    for _ in range(_MAX_TOOL_ITERATIONS):
        stream = _svc.stream_chat_complete(
            current, controls, tools=all_t, tool_choice=effective_choice, json_mode=json_mode
        )
        effective_choice = "auto"

        raw_chunks = []
        is_tool_call = False
        is_streaming_to_client = True

        for chunk in stream:
            raw_chunks.append(chunk)

            if chunk.get("tool_call_chunks"):
                is_tool_call = True
                is_streaming_to_client = False
                continue

            if is_streaming_to_client:
                yield from _sse_stream(iter([chunk]), model_id, cid, append_done=False)

        tool_calls = _collect_stream_tool_calls(raw_chunks)

        if not tool_calls:
            # Final text turn is done
            yield "data: [DONE]\n\n"
            return

        if _has_non_builtin(tool_calls):
            # Contains client-managed tools — stream the buffered tool call chunks to caller
            yield from _sse_stream(iter(raw_chunks), model_id, cid, append_done=False)
            yield "data: [DONE]\n\n"
            return

        # Silently execute built-in tool call and repeat agentic loop
        text = "".join(c.get("delta", "") for c in raw_chunks) or None
        current.append({
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        })
        current.extend(_execute_tool_calls(tool_calls))

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
    client_tools = [t.model_dump() for t in request.tools] if request.tools else None
    tool_choice = request.tool_choice
    json_mode = request.response_format.type in ("json_object", "json_schema")

    if request.stream:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        return StreamingResponse(
            _agentic_stream(messages, controls, client_tools, tool_choice, model_id, cid, json_mode=json_mode),
            media_type="text/event-stream",
        )

    result = _run_agentic_chat(messages, controls, client_tools, tool_choice, json_mode=json_mode)
    tool_calls_out: list[ToolCall] | None = None
    if result.get("tool_calls"):
        tool_calls_out = [ToolCall(**tc) for tc in result["tool_calls"]]
    active_builtin_names = [s["function"]["name"] for s in BUILTIN_REGISTRY.schemas()]
    return ChatCompletionResponse(
        id=result["id"],
        created=result["created"],
        model=model_id,
        choices=[
            OAIChoice(
                message=OAIMessage(content=result["output"], tool_calls=tool_calls_out),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=OAIUsage(**result["usage"]),
        system_tools=active_builtin_names,
    )

