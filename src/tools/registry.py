"""Tool registry: maps tool names → (OpenAI schema, callable)."""
from __future__ import annotations

import json
from typing import Any, Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

    def register(self, schema: dict[str, Any], fn: Callable[..., Any]) -> None:
        name: str = schema["function"]["name"]
        self._tools[name] = (schema, fn)

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        return [schema for schema, _ in self._tools.values()]

    def execute(self, name: str, arguments: str) -> str:
        """Parse *arguments* JSON, call the registered function, return JSON string."""
        _, fn = self._tools[name]
        kwargs: dict[str, Any] = json.loads(arguments) if arguments.strip() else {}
        result = fn(**kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


BUILTIN_REGISTRY: ToolRegistry = ToolRegistry()
