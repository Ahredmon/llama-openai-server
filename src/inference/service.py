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
    max_tokens: int = Field(default=256, ge=1, le=131072)
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


def _query_vram_mb() -> int | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        for card_key, card_data in data.items():
            if not card_key.startswith("card"):
                continue
            for k, v in card_data.items():
                if "total" in k.lower() and "vram" in k.lower():
                    return int(v) // (1024 * 1024)
    except Exception:
        pass
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
# Backend protocol
# ---------------------------------------------------------------------------


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
        self, messages: list[dict], controls: GenerationControls
    ) -> tuple[str, dict[str, int], str]:
        raise NotImplementedError

    def stream_chat_complete_messages(
        self, messages: list[dict], controls: GenerationControls
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
        self, messages: list[dict], controls: GenerationControls
    ) -> tuple[str, dict[str, int], str]:
        text = f"[mock] {len(messages)} messages"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return text, usage, "stop"

    def stream_chat_complete_messages(
        self, messages: list[dict], controls: GenerationControls
    ) -> Generator[dict[str, Any], None, None]:
        for word in f"[mock] {len(messages)} messages".split():
            yield {"delta": f"{word} ", "done": False}
        yield {"delta": "", "done": True, "finish_reason": "stop"}


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
    ) -> None:
        self.model_id = model_id
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with ROCm flags:\n"
                '  CMAKE_ARGS="-DGGML_HIPBLAS=on -DCMAKE_HIP_ARCHITECTURES=gfx1201" '
                "FORCE_CMAKE=1 pip install llama-cpp-python --no-binary llama-cpp-python"
            ) from exc

        n_threads_batch = n_threads_batch or max(1, n_threads // 2)
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
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
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
        self, messages: list[dict], controls: GenerationControls
    ) -> tuple[str, dict[str, int], str]:
        response = self._llm.create_chat_completion(
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
        choice = response["choices"][0]
        text = str(choice.get("message", {}).get("content", "") or "")
        usage = response.get("usage", {})
        return (
            text,
            {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            },
            str(choice.get("finish_reason", "stop")),
        )

    def stream_chat_complete_messages(
        self, messages: list[dict], controls: GenerationControls
    ) -> Generator[dict[str, Any], None, None]:
        stream = self._llm.create_chat_completion(
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
        for item in stream:
            choice = item["choices"][0]
            finish_reason = choice.get("finish_reason")
            yield {
                "delta": str(choice.get("delta", {}).get("content", "") or ""),
                "done": finish_reason is not None,
                "finish_reason": str(finish_reason) if finish_reason else None,
            }


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
    ) -> dict[str, Any]:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")

        def _work() -> tuple[str, dict[str, int], str]:
            return self._backend.chat_complete_messages(messages, controls)  # type: ignore[union-attr]

        output, usage, finish_reason = self._executor.submit(_work).result()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "model": self.settings.model_id,
            "output": output,
            "finish_reason": finish_reason,
            "usage": usage,
            "created": int(time.time()),
        }

    def stream_chat_complete(
        self,
        messages: list[dict],
        controls: GenerationControls,
    ) -> Generator[dict[str, Any], None, None]:
        if not self._backend:
            raise RuntimeError("Inference backend not initialized")

        def _work() -> list[dict[str, Any]]:
            return list(self._backend.stream_chat_complete_messages(messages, controls))  # type: ignore[union-attr]

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
