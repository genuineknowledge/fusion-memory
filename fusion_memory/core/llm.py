from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib import request


class LLMClient(Protocol):
    def structured(self, prompt: str, schema: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
        ...


class StaticLLMClient:
    """Test/dry-run LLM client returning a fixed structured payload."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def structured(self, prompt: str, schema: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "schema": schema, "input": input})
        return self.response


class OpenAICompatibleLLMClient:
    """Dependency-free structured LLM client for OpenAI-compatible endpoints.

    The endpoint is expected to accept chat-completions shaped JSON and return
    either a JSON object in `choices[0].message.content` or an already
    structured object under `structured`.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str = "local-structured-extractor",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.calls: list[dict[str, Any]] = []
        self.version = f"openai-compatible:{model}"

    def structured(self, prompt: str, schema: dict[str, Any], input: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON matching the provided schema.",
                },
                {
                    "role": "user",
                    "content": json.dumps({"prompt": prompt, "schema": schema, "input": input}, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        data = _post_json(self.endpoint, payload, api_key=self.api_key, timeout_seconds=self.timeout_seconds)
        latency_ms = (time.perf_counter() - started) * 1000
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "input": input,
                "model": self.model,
                "latency_ms": latency_ms,
                "usage": data.get("usage", {}),
            }
        )
        return _extract_structured_response(data)


def _post_json(endpoint: str, payload: dict[str, Any], *, api_key: str | None, timeout_seconds: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("LLM endpoint must return a JSON object")
    return data


def _extract_structured_response(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("structured"), dict):
        return data["structured"]
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
    if any(key in data for key in ["facts", "events", "relations", "answer", "matched"]):
        return data
    raise ValueError("LLM endpoint did not return a structured JSON object")
