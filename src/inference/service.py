from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generation controls
# ---------------------------------------------------------------------------


class GenerationControls(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    max_tokens: int = Field(default=2048, ge=1, le=131072)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: list[str] = Field(default_factory=list, max_length=16)
    seed: int | None = None


# ---------------------------------------------------------------------------
# VRAM / n_ctx auto-detection
# ---------------------------------------------------------------------------

_CTX_TIERS: list[tuple[int, int]] = [
    (6 * 1024, 4096),
    (8 * 1024, 8192),
    (12 * 1024, 16384),
    (16 * 1024, 32768),
    (24 * 1024, 65536),
    (int(1e9), 131072),
]
_CTX_FALLBACK = 4096


# Candidate paths for rocm-smi; /opt/rocm/bin is not always on PATH under systemd.
_ROCM_SMI_CANDIDATES = [
    "rocm-smi",
    "/opt/rocm/bin/rocm-smi",
    "/usr/bin/rocm-smi",
]


def _query_vram_mb() -> int | None:
    for smi in _ROCM_SMI_CANDIDATES:
        try:
            result = subprocess.run(
                [smi, "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                continue
            data = json.loads(result.stdout)
            for card_key, card_data in data.items():
                if not card_key.startswith("card"):
                    continue
                for k, v in card_data.items():
                    # Match "VRAM Total Memory (B)" but NOT "Used Memory".
                    kl = k.lower()
                    if "vram" in kl and "total" in kl and "used" not in kl:
                        return int(v) // (1024 * 1024)
        except Exception:
            continue
    return None


def _auto_detect_n_ctx() -> int:
    vram_mb = _query_vram_mb()
    if vram_mb is None:
        logger.warning("n_ctx_auto_detect: could not query VRAM; falling back to %d", _CTX_FALLBACK)
        return _CTX_FALLBACK
    for threshold_mb, ctx in _CTX_TIERS:
        if vram_mb < threshold_mb:
            logger.info("n_ctx_auto_detect: %.1f GB VRAM -> n_ctx=%d", vram_mb / 1024, ctx)
            return ctx
    logger.info("n_ctx_auto_detect: %.1f GB VRAM -> n_ctx=%d", vram_mb / 1024, _CTX_TIERS[-1][1])
    return _CTX_TIERS[-1][1]


# ---------------------------------------------------------------------------
# GBNF grammar for JSON-constrained output
# ---------------------------------------------------------------------------

_JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\\x7F\x00-\x1F\"] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))? ws

ws ::= ([ \t\n] ws)?
""".strip()


# ---------------------------------------------------------------------------
# Gemma 4 native tool-call parser
# ---------------------------------------------------------------------------

# Gemma 4 outputs tool calls in one of two formats:
#   1. <|tool_call>call:FNAME{key:<|"|>value<|"|>}<tool_call|>  (bare keys + special quote tokens)
#   2. <|tool_call>call:FNAME{{"key": "value"}}<tool_call|>     (double-brace JSON)
# llama-cpp-python's "gemma" chat handler does not parse either back into
# structured tool_calls — it just returns the raw text as content.
# We detect and parse both variants ourselves.

_GEMMA_TC_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z0-9_]+)(\{\{.*?\}\}|\{[^}]*\})<tool_call\|>",
    re.DOTALL,
)
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")


def _parse_gemma_tool_calls(text: str) -> list[dict] | None:
    """
    Parse Gemma 4's native <|tool_call>call:FNAME{...}<tool_call|> syntax
    into OpenAI-compatible tool_calls dicts.  Returns None if no matches.

    Two argument formats are handled:
      - Double-brace JSON: {{"key": "value"}} — strip outer braces to get valid JSON
      - Single-brace with special quote tokens: {key:<|"|>value<|"|>} — replace tokens
        and quote bare keys
    """
    matches = _GEMMA_TC_RE.findall(text)
    if not matches:
        return None
    result = []
    for fn_name, raw_args in matches:
        if raw_args.startswith("{{"):
            # Double-brace format: {{...}} → strip one layer of braces → valid JSON
            args_str = raw_args[1:-1]
        else:
            # Single-brace format with Gemma's special quote token <|"|>
            args_str = raw_args.replace('<|"|>', '"')
            # Quote any bare (unquoted) object keys to make it valid JSON
            args_str = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', args_str)
        try:
            args_obj = json.loads(args_str)
            args_json = json.dumps(args_obj)
        except json.JSONDecodeError:
            args_json = "{}"
        result.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": args_json},
        })
    return result or None


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vision chat-handler factory
# ---------------------------------------------------------------------------

_HANDLER_MAP: dict[str, str] = {
    "llava15": "Llava15ChatHandler",
    "llava16": "Llava16ChatHandler",
    "moondream": "MoondreamChatHandler",
    "qwen2vl": "Qwen25VLChatHandler",
    "gemma3": "Gemma3ChatHandler",
    "minicpmv": "MiniCPMv26ChatHandler",
}


def _make_chat_handler(clip_model_path: str, handler_hint: str, model_path: str) -> Any:
    """
    Create the llama-cpp-python vision chat handler appropriate for the loaded model.

    ``handler_hint`` can be any key in _HANDLER_MAP or ``"auto"`` (detect from
    the model filename).  For Gemma 3/4 models we use a local subclass that
    keeps the full mtmd image-encoding pipeline from Llava15ChatHandler but
    swaps in Gemma's native ``<start_of_turn>`` / ``<end_of_turn>`` chat
    template so that thinking tokens and tool-call syntax are preserved.
    """
    try:
        from llama_cpp import llama_chat_format
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed") from exc

    # ---------------------------------------------------------------------------
    # Gemma 4 vision handler: inherits all mtmd image-encoding from Llava15 but
    # uses Gemma's native chat template instead of the LLaVA USER:/ASSISTANT:
    # format.  This lets <thinking> tokens and native tool-call syntax work.
    # All Gemma-specific logic (template, tool injection, tool-call parsing)
    # lives here so it is not fractured across backend methods.
    # ---------------------------------------------------------------------------
    class _Gemma4VisionChatHandler(llama_chat_format.Llava15ChatHandler):
        # Do not inject a LLaVA system prompt – Gemma handles system turns as
        # a regular <start_of_turn>user turn prepended to the conversation.
        DEFAULT_SYSTEM_MESSAGE = None

        # Gemma 4 chat template.  Image URLs are placed inline where the image
        # should appear; Llava15ChatHandler.__call__ replaces them with the
        # mtmd media-marker token before tokenising.
        CHAT_FORMAT = (
            "{% for message in messages %}"
            # System message → wrapped in a user turn (Gemma convention)
            "{% if message.role == 'system' %}"
            "<start_of_turn>user\n{{ message.content }}<end_of_turn>\n"
            "{% endif %}"
            # User message
            "{% if message.role == 'user' %}"
            "<start_of_turn>user\n"
            "{% if message.content is iterable %}"
            # Emit image URLs first so they appear before the text
            "{% for content in message.content %}"
            "{% if content.type == 'image_url' and content.image_url is string %}"
            "{{ content.image_url }}"
            "{% endif %}"
            "{% if content.type == 'image_url' and content.image_url is mapping %}"
            "{{ content.image_url.url }}"
            "{% endif %}"
            "{% endfor %}"
            # Then emit the text parts
            "{% for content in message.content %}"
            "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "{% if message.content is string %}{{ message.content }}{% endif %}"
            "<end_of_turn>\n"
            "{% endif %}"
            # Assistant message
            "{% if message.role == 'assistant' and message.content is not none %}"
            "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        )

        def __call__(self, *, llama, messages, tools=None, tool_choice=None, **kwargs):
            # ------------------------------------------------------------------
            # 0. Normalise tool-history turns into Gemma-native text so the
            #    template can render them.  Without this, assistant tool_calls
            #    messages and tool-result messages are silently dropped from
            #    the rendered prompt — the model never sees its previous call
            #    or the result, so it re-calls the same tool every turn.
            #
            #    Gemma 4 convention:
            #      assistant tool-call  → <|tool_call>call:NAME{...}<tool_call|>
            #      tool result          → user turn with <tool_response>…</tool_response>
            # ------------------------------------------------------------------
            normalised: list[dict] = []
            for m in messages:
                role = m.get("role")
                if role == "assistant" and not m.get("content") and m.get("tool_calls"):
                    # Convert OpenAI-format tool_calls into Gemma native tokens
                    parts = []
                    for tc in m["tool_calls"]:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args = fn.get("arguments", "{}")
                        parts.append(f"<|tool_call>call:{name}{args}<tool_call|>")
                    normalised.append({"role": "assistant", "content": "".join(parts)})
                elif role == "tool":
                    # Tool results become a user turn with <tool_response> wrapper
                    normalised.append({
                        "role": "user",
                        "content": f"<tool_response>\n{m.get('content', '')}\n</tool_response>",
                    })
                else:
                    normalised.append(m)
            messages = normalised

            # ------------------------------------------------------------------
            # 1. Inject tool definitions into the system message so the model
            #    knows what tools exist (required for native auto tool-calling).
            # ------------------------------------------------------------------
            if tools:
                tool_defs = json.dumps(
                    [t["function"] for t in tools],
                    ensure_ascii=False,
                    indent=2,
                )
                tool_preamble = (
                    "You have access to the following tools:\n"
                    + tool_defs
                    + "\n\nWhen you want to call a tool, use Gemma's native format:\n"
                    "<|tool_call>call:FUNCTION_NAME{\"arg\": \"value\"}<tool_call|>"
                )
                messages = list(messages)
                sys_idx = next(
                    (i for i, m in enumerate(messages) if m.get("role") == "system"),
                    None,
                )
                if sys_idx is not None:
                    m = messages[sys_idx]
                    messages[sys_idx] = {
                        **m,
                        "content": m["content"] + "\n\n" + tool_preamble,
                    }
                else:
                    messages.insert(0, {"role": "system", "content": tool_preamble})

            # ------------------------------------------------------------------
            # 2. Delegate to parent — handles image encoding, template rendering,
            #    prompt tokenisation, and completion.
            # ------------------------------------------------------------------
            stream = kwargs.get("stream", False)
            result = super().__call__(
                llama=llama,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

            if not tools:
                return result

            # ------------------------------------------------------------------
            # 3. Post-process: parse Gemma 4 native tool-call tokens so the
            #    response is OpenAI-compatible regardless of llama-cpp-python
            #    version.  This keeps all Gemma behaviour in one place.
            # ------------------------------------------------------------------
            if not stream:
                # Non-streaming: result is a CreateChatCompletionResponse dict.
                for choice in result.get("choices", []):
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    if msg.get("tool_calls") is None and content:
                        parsed = _parse_gemma_tool_calls(content)
                        if parsed:
                            msg["tool_calls"] = parsed
                            msg["content"] = None
                            choice["finish_reason"] = "tool_calls"
                return result

            # Streaming: result is an iterator of CreateChatCompletionStreamResponse
            # dicts.  Buffer to detect Gemma tool-call tokens in the text stream.
            def _postprocess_stream(chunks_iter):
                chunks = list(chunks_iter)
                # If llama-cpp-python already emitted structured tool_calls in
                # the delta, pass through as-is.
                if any(
                    bool(c.get("delta", {}).get("tool_calls"))
                    for item in chunks
                    for c in item.get("choices", [])
                ):
                    yield from chunks
                    return

                full_text = "".join(
                    c.get("delta", {}).get("content", "") or ""
                    for item in chunks
                    for c in item.get("choices", [])
                )
                parsed = _parse_gemma_tool_calls(full_text)
                if not parsed:
                    yield from chunks
                    return

                # Emit one chunk per tool call with structured tool_calls delta.
                first = chunks[0] if chunks else {}
                last = chunks[-1] if chunks else {}
                for i, tc in enumerate(parsed):
                    yield {
                        "id": first.get("id", ""),
                        "object": "chat.completion.chunk",
                        "created": first.get("created", 0),
                        "model": first.get("model", ""),
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant" if i == 0 else None,
                                "content": None,
                                "tool_calls": [{
                                    "index": i,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    },
                                }],
                            },
                            "logprobs": None,
                            "finish_reason": None,
                        }],
                    }
                yield {
                    "id": last.get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": last.get("created", 0),
                    "model": last.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "tool_calls",
                    }],
                }

            return _postprocess_stream(result)

    hint = handler_hint.lower().strip()
    if hint == "auto":
        name = Path(model_path).name.lower()
        if "gemma" in name:
            hint = "gemma3"
        elif "llava-1.6" in name or "llava16" in name:
            hint = "llava16"
        elif "moondream" in name:
            hint = "moondream"
        elif "qwen2" in name and "vl" in name:
            hint = "qwen2vl"
        elif "minicpm" in name:
            hint = "minicpmv"
        else:
            hint = "llava15"

    # Gemma 3/4: always use our local handler so we keep both image encoding
    # AND the native thinking/tool-call template regardless of llama-cpp build.
    if hint in ("gemma3", "gemma4"):
        logger.info(
            "vision_handler_init",
            extra={"handler": "_Gemma4VisionChatHandler", "clip_model_path": clip_model_path},
        )
        return _Gemma4VisionChatHandler(clip_model_path=clip_model_path, verbose=False)

    cls_name = _HANDLER_MAP.get(hint, "Llava15ChatHandler")
    handler_cls = getattr(llama_chat_format, cls_name, None)
    if handler_cls is None:
        # The exact handler isn't available in this llama-cpp-python build.
        # Fall back to Llava15ChatHandler so the mmproj still encodes images;
        # without any handler, image_url parts in messages are silently dropped.
        fallback_cls = getattr(llama_chat_format, "Llava15ChatHandler", None)
        if fallback_cls is None:
            logger.warning(
                "vision_handler_not_found",
                extra={"requested": cls_name, "note": "No handler available; images will not be processed"},
            )
            return None
        logger.warning(
            "vision_handler_fallback",
            extra={"requested": cls_name, "fallback": "Llava15ChatHandler",
                   "note": "Using Llava15 for image encoding; tool calling may be limited"},
        )
        handler_cls = fallback_cls

    logger.info("vision_handler_init", extra={"handler": handler_cls.__name__, "clip_model_path": clip_model_path})
    return handler_cls(clip_model_path=clip_model_path, verbose=False)


class _Backend:
    name = "base"

    def complete(
        self, prompt: str, controls: GenerationControls, json_mode: bool = False
    ) -> tuple[str, dict[str, int], str]:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, controls: GenerationControls, json_mode: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        raise NotImplementedError

    def chat_complete_messages(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> tuple[str | None, list[dict] | None, dict[str, int], str]:
        raise NotImplementedError

    def stream_chat_complete_messages(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock backend (used when ALLOW_MOCK_BACKEND=true or in tests)
# ---------------------------------------------------------------------------


class _MockBackend(_Backend):
    name = "mock"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def complete(
        self, prompt: str, controls: GenerationControls, json_mode: bool = False
    ) -> tuple[str, dict[str, int], str]:
        output = f"[mock:{self.model_id}] {prompt[: controls.max_tokens]}"
        if json_mode:
            output = json.dumps({"mock": True, "model": self.model_id})
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(output.split()))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return output, usage, "stop"

    def stream_complete(
        self, prompt: str, controls: GenerationControls, json_mode: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        words = f"[mock:{self.model_id}] {prompt}".split()
        for word in words[: controls.max_tokens]:
            yield {"delta": f"{word} ", "done": False}
        yield {"delta": "", "done": True, "finish_reason": "stop"}

    def chat_complete_messages(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> tuple[str | None, list[dict] | None, dict[str, int], str]:
        text = f"[mock] {len(messages)} messages"
        if json_mode:
            text = json.dumps({"mock": True, "messages": len(messages)})
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return text, None, usage, "stop"

    def stream_chat_complete_messages(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        for word in f"[mock] {len(messages)} messages".split():
            yield {"delta": f"{word} ", "tool_call_chunks": None, "done": False}
        yield {"delta": "", "tool_call_chunks": None, "done": True, "finish_reason": "stop"}


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------


class _LlamaCppBackend(_Backend):
    name = "llama.cpp"

    def __init__(
        self,
        model_id: str,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_threads_batch: int | None,
        n_batch: int,
        n_ubatch: int,
        n_gpu_layers: int,
        mlock: bool,
        offload_kqv: bool,
        flash_attn: bool,
        f16_kv: bool,
        type_k: int | None = None,
        type_v: int | None = None,
        clip_model_path: str | None = None,
        clip_chat_handler: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.vision_enabled = clip_model_path is not None
        # Resolve KV cache type: explicit type_k/v override f16_kv flag.
        # GGML_TYPE_F16=1, GGML_TYPE_Q8_0=8, GGML_TYPE_Q4_0=2
        _type_k = type_k if type_k is not None else (1 if f16_kv else 8)
        _type_v = type_v if type_v is not None else (1 if f16_kv else 8)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with ROCm flags:\n"
                '  CMAKE_ARGS="-DGGML_HIP=on -DCMAKE_HIP_ARCHITECTURES=gfx1201" '
                "FORCE_CMAKE=1 pip install llama-cpp-python --no-binary llama-cpp-python"
            ) from exc

        n_threads_batch = n_threads_batch or max(1, n_threads // 2)

        chat_handler = None
        if clip_model_path:
            chat_handler = _make_chat_handler(clip_model_path, clip_chat_handler, model_path)

        logger.info(
            "llama_cpp_backend_init",
            extra={
                "model_path": model_path,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "mlock": mlock,
                "offload_kqv": offload_kqv,
                "flash_attn": flash_attn,
                "f16_kv": f16_kv,
                "type_k": _type_k,
                "type_v": _type_v,
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
                "vision": self.vision_enabled,
            },
        )

        # Try most-capable init first; fall back progressively for older
        # llama-cpp-python versions that lack certain keyword arguments.
        for kwargs in [
            dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                n_gpu_layers=n_gpu_layers,
                mlock=mlock,
                offload_kqv=offload_kqv,
                flash_attn=flash_attn,
                f16_kv=f16_kv,
                type_k=_type_k,
                type_v=_type_v,
                chat_handler=chat_handler,
                verbose=False,
            ),
            dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                n_gpu_layers=n_gpu_layers,
                mlock=mlock,
                offload_kqv=offload_kqv,
                flash_attn=flash_attn,
                f16_kv=f16_kv,
                chat_handler=chat_handler,
                verbose=False,
            ),
            dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                n_gpu_layers=n_gpu_layers,
                mlock=mlock,
                chat_handler=chat_handler,
                verbose=False,
            ),
            dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                n_gpu_layers=n_gpu_layers,
                mlock=mlock,
                verbose=False,
            ),
        ]:
            try:
                self._llm = Llama(**kwargs)
                break
            except TypeError as exc:
                logger.warning("llama_init_fallback", extra={"error": str(exc)})
        else:
            raise RuntimeError("Could not initialise llama.cpp backend after multiple attempts")

    def complete(
        self, prompt: str, controls: GenerationControls, json_mode: bool = False
    ) -> tuple[str, dict[str, int], str]:
        from llama_cpp import LlamaGrammar

        grammar = LlamaGrammar.from_string(_JSON_GBNF) if json_mode else None
        response = self._llm.create_completion(
            prompt=prompt,
            max_tokens=controls.max_tokens,
            temperature=controls.temperature,
            top_p=controls.top_p,
            top_k=controls.top_k,
            repeat_penalty=controls.repeat_penalty,
            frequency_penalty=controls.frequency_penalty or 0.0,
            presence_penalty=controls.presence_penalty or 0.0,
            stop=controls.stop or None,
            seed=controls.seed,
            stream=False,
            grammar=grammar,
        )
        choice = response["choices"][0]
        usage = response.get("usage", {})
        return (
            str(choice.get("text", "")),
            {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
            str(choice.get("finish_reason", "stop")),
        )

    def stream_complete(
        self, prompt: str, controls: GenerationControls, json_mode: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        from llama_cpp import LlamaGrammar

        grammar = LlamaGrammar.from_string(_JSON_GBNF) if json_mode else None
        stream = self._llm.create_completion(
            prompt=prompt,
            max_tokens=controls.max_tokens,
            temperature=controls.temperature,
            top_p=controls.top_p,
            top_k=controls.top_k,
            repeat_penalty=controls.repeat_penalty,
            frequency_penalty=controls.frequency_penalty or 0.0,
            presence_penalty=controls.presence_penalty or 0.0,
            stop=controls.stop or None,
            seed=controls.seed,
            stream=True,
            grammar=grammar,
        )
        for item in stream:
            choice = item["choices"][0]
            finish_reason = choice.get("finish_reason")
            yield {
                "delta": str(choice.get("text", "")),
                "done": finish_reason is not None,
                "finish_reason": str(finish_reason) if finish_reason else None,
            }

    def chat_complete_messages(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> tuple[str | None, list[dict] | None, dict[str, int], str]:
        kwargs: dict[str, Any] = dict(
            messages=messages,
            max_tokens=controls.max_tokens,
            temperature=controls.temperature,
            top_p=controls.top_p,
            top_k=controls.top_k,
            repeat_penalty=controls.repeat_penalty,
            frequency_penalty=controls.frequency_penalty or 0.0,
            presence_penalty=controls.presence_penalty or 0.0,
            stop=controls.stop or None,
            seed=controls.seed,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        elif json_mode:
            from llama_cpp import LlamaGrammar
            kwargs["grammar"] = LlamaGrammar.from_string(_JSON_GBNF)
        response = self._llm.create_chat_completion(**kwargs)
        choice = response["choices"][0]
        msg = choice.get("message", {})
        text: str | None = msg.get("content") or None
        tool_calls: list[dict] | None = msg.get("tool_calls") or None
        finish_reason = str(choice.get("finish_reason", "stop"))
        # Gemma 4 outputs tool calls in its own format; llama-cpp-python won't
        # parse them automatically — detect and convert them here.
        if tool_calls is None and text is not None:
            parsed = _parse_gemma_tool_calls(text)
            if parsed:
                tool_calls = parsed
                text = None
                finish_reason = "tool_calls"
        usage = response.get("usage", {})
        return (
            text,
            tool_calls,
            {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
            finish_reason,
        )

    def stream_chat_complete_messages(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        kwargs: dict[str, Any] = dict(
            messages=messages,
            max_tokens=controls.max_tokens,
            temperature=controls.temperature,
            top_p=controls.top_p,
            top_k=controls.top_k,
            repeat_penalty=controls.repeat_penalty,
            frequency_penalty=controls.frequency_penalty or 0.0,
            presence_penalty=controls.presence_penalty or 0.0,
            stop=controls.stop or None,
            seed=controls.seed,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        elif json_mode:
            from llama_cpp import LlamaGrammar
            kwargs["grammar"] = LlamaGrammar.from_string(_JSON_GBNF)
        stream = self._llm.create_chat_completion(**kwargs)

        if not tools:
            # No tool calling involved — pure streaming passthrough.
            for item in stream:
                choice = item["choices"][0]
                finish_reason = choice.get("finish_reason")
                delta = choice.get("delta", {})
                yield {
                    "delta": str(delta.get("content", "") or ""),
                    "tool_call_chunks": delta.get("tool_calls"),
                    "done": finish_reason is not None,
                    "finish_reason": str(finish_reason) if finish_reason else None,
                }
            return

        # Tools were requested: buffer the full stream so we can detect Gemma 4's
        # native <|tool_call>...<tool_call|> syntax (llama-cpp-python streams it
        # as plain content, not as tool_call_chunks).
        raw_chunks: list[dict[str, Any]] = []
        for item in stream:
            choice = item["choices"][0]
            finish_reason = choice.get("finish_reason")
            delta = choice.get("delta", {})
            raw_chunks.append({
                "delta": str(delta.get("content", "") or ""),
                "tool_call_chunks": delta.get("tool_calls"),
                "done": finish_reason is not None,
                "finish_reason": str(finish_reason) if finish_reason else None,
            })

        # If llama-cpp-python already gave us structured tool_call_chunks, pass
        # the buffer through unchanged.
        if any(c.get("tool_call_chunks") for c in raw_chunks):
            yield from raw_chunks
            return

        # Check whether the accumulated text is a Gemma 4 native tool call.
        full_text = "".join(c["delta"] for c in raw_chunks)
        parsed = _parse_gemma_tool_calls(full_text)
        if parsed:
            for i, tc in enumerate(parsed):
                yield {
                    "delta": "",
                    "tool_call_chunks": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }],
                    "done": False,
                    "finish_reason": None,
                }
            yield {"delta": "", "tool_call_chunks": None, "done": True, "finish_reason": "tool_calls"}
            return

        # Plain text response — replay the buffered chunks.
        yield from raw_chunks


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------


class InferenceService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._backend: _Backend | None = None
        self._ready = False
        self._model_path: str | None = None
        # ALL llama.cpp / HIP calls are dispatched to this single worker thread.
        # The HIP GPU context is created here and must never cross threads.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llama-worker"
        )

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self) -> None:
        if getattr(self.settings, "allow_mock_backend", False):
            self._backend = _MockBackend(model_id=self.settings.model_id)
            self._ready = True
            self._model_path = "mock"
            return

        self._validate_rocm()
        model_path = self._resolve_model_path()
        clip_path = self._resolve_clip_path()
        n_ctx = self.settings.n_ctx if self.settings.n_ctx > 0 else _auto_detect_n_ctx()

        def _init() -> _Backend:
            return _LlamaCppBackend(
                model_id=self.settings.model_id,
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=self.settings.n_threads,
                n_threads_batch=self.settings.n_threads_batch,
                n_batch=self.settings.n_batch,
                n_ubatch=self.settings.n_ubatch,
                n_gpu_layers=self.settings.n_gpu_layers,
                mlock=self.settings.mlock,
                offload_kqv=self.settings.offload_kqv,
                flash_attn=self.settings.flash_attn,
                f16_kv=self.settings.f16_kv,
                type_k=getattr(self.settings, "type_k", None),
                type_v=getattr(self.settings, "type_v", None),
                clip_model_path=str(clip_path) if clip_path else None,
                clip_chat_handler=self.settings.clip_chat_handler,
            )

        self._backend = self._executor.submit(_init).result()
        self._model_path = str(model_path)
        self._ready = True

    def model_metadata(self) -> dict:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")
        return {
            "id": self.settings.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }

    def complete(
        self,
        prompt: str,
        controls: GenerationControls,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")
        full_prompt = _build_prompt(system_prompt=system_prompt, prompt=prompt)

        def _work() -> tuple[str, dict[str, int], str]:
            return self._backend.complete(full_prompt, controls, json_mode=json_mode)  # type: ignore[union-attr]

        output, usage, finish_reason = self._executor.submit(_work).result()
        if json_mode:
            output = _clean_json_output(output)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "model": self.settings.model_id,
            "output": output,
            "finish_reason": finish_reason,
            "usage": usage,
            "created": int(time.time()),
        }

    def stream_complete(
        self,
        prompt: str,
        controls: GenerationControls,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")
        full_prompt = _build_prompt(system_prompt=system_prompt, prompt=prompt)

        def _work() -> list[dict[str, Any]]:
            return list(self._backend.stream_complete(full_prompt, controls, json_mode=json_mode))  # type: ignore[union-attr]

        yield from self._executor.submit(_work).result()

    def chat_complete(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")

        def _work() -> tuple[str | None, list[dict] | None, dict[str, int], str]:
            return self._backend.chat_complete_messages(messages, controls, tools, tool_choice, json_mode)  # type: ignore[union-attr]

        output, tool_calls, usage, finish_reason = self._executor.submit(_work).result()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "model": self.settings.model_id,
            "output": output,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": usage,
            "created": int(time.time()),
        }

    def stream_chat_complete(
        self,
        messages: list[dict],
        controls: GenerationControls,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        json_mode: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")

        def _work() -> list[dict[str, Any]]:
            return list(self._backend.stream_chat_complete_messages(messages, controls, tools, tool_choice, json_mode))  # type: ignore[union-attr]

        yield from self._executor.submit(_work).result()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_rocm(self) -> None:
        if not getattr(self.settings, "rocm_required", True):
            return
        has_rocm = any(
            [
                "ROCM_HOME" in os.environ,
                "HIP_PATH" in os.environ,
                Path("/opt/rocm").exists(),
            ]
        )
        if not has_rocm:
            raise RuntimeError(
                "ROCm-only mode is enabled but ROCm was not detected. "
                "Set ROCM_HOME/HIP_PATH or install ROCm under /opt/rocm. "
                "Set ROCM_REQUIRED=false to bypass this check."
            )
        if self.settings.n_gpu_layers <= 0:
            raise RuntimeError("N_GPU_LAYERS must be > 0 in ROCm-only mode.")
        # Verify the llama-cpp-python binary was compiled with GPU offload support.
        # A CPU wheel passes the OS-level ROCm check above but silently falls back
        # to CPU inference. Fail fast here so the error is obvious.
        try:
            import llama_cpp
            if not llama_cpp.llama_supports_gpu_offload():
                raise RuntimeError(
                    "llama-cpp-python is installed but was NOT compiled with GPU "
                    "offload support (llama_supports_gpu_offload() returned False). "
                    "Rebuild from source with ROCm flags:\n"
                    "  bash scripts/install-rocm-llama.sh\n"
                    "Set ROCM_REQUIRED=false to run on CPU anyway."
                )
        except ImportError:
            pass  # ImportError is reported later during backend init

    def _resolve_model_path(self) -> Path:
        if self.settings.model_path:
            p = Path(self.settings.model_path)
            if p.exists():
                return p
            raise RuntimeError(f"MODEL_PATH set but file not found: {p}")

        if self.settings.hf_model_filename:
            local = Path(self.settings.models_dir) / self.settings.hf_model_filename
            if local.exists():
                return local

        if self.settings.hf_repo_id and self.settings.hf_model_filename:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is required for model auto-download: "
                    "pip install huggingface_hub"
                ) from exc
            downloaded = hf_hub_download(
                repo_id=self.settings.hf_repo_id,
                filename=self.settings.hf_model_filename,
                revision=self.settings.hf_revision,
                local_dir=self.settings.models_dir,
            )
            return Path(downloaded)

        raise RuntimeError(
            "No model file found. Set MODEL_PATH or configure "
            "HF_REPO_ID + HF_MODEL_FILENAME."
        )

    def _resolve_clip_path(self) -> Path | None:
        """Resolve the mmproj/CLIP model path, downloading from HF if needed.

        Returns None when no clip model is configured (text-only mode).
        """
        if self.settings.clip_model_path:
            p = Path(self.settings.clip_model_path)
            if p.exists():
                return p
            raise RuntimeError(f"CLIP_MODEL_PATH set but file not found: {p}")

        if self.settings.hf_clip_filename:
            local = Path(self.settings.models_dir) / self.settings.hf_clip_filename
            if local.exists():
                return local

        if self.settings.hf_clip_repo_id and self.settings.hf_clip_filename:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is required for clip model auto-download: "
                    "pip install huggingface_hub"
                ) from exc
            downloaded = hf_hub_download(
                repo_id=self.settings.hf_clip_repo_id,
                filename=self.settings.hf_clip_filename,
                revision=self.settings.hf_revision,
                local_dir=self.settings.models_dir,
            )
            return Path(downloaded)

        logger.info("vision_disabled: no CLIP_MODEL_PATH configured")
        return None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_prompt(system_prompt: str | None, prompt: str) -> str:
    if not system_prompt:
        return prompt
    return f"<SYSTEM>\n{system_prompt}\n</SYSTEM>\n\n<USER>\n{prompt}\n</USER>\n\n<ASSISTANT>\n"


_MD_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _clean_json_output(raw: str) -> str:
    text = raw.strip()
    fence = _MD_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            text = text[i:]
            break
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("}", "]"):
            text = text[: i + 1]
            break
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output is not valid JSON: {exc}") from exc
    return json.dumps(parsed, ensure_ascii=True)
