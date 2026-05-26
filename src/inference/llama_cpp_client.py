from __future__ import annotations

from typing import Any

import httpx


class LlamaCppClient:
    """Small client for llama.cpp OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001/v1",
        api_key: str = "not-needed",
        model: str = "discord-qwen-local",
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        enable_thinking: bool | None = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return _extract_message_text(data)


def _extract_message_text(data: dict[str, Any]) -> str:
    choice = data["choices"][0]
    message = choice.get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content:
        return content

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        return reasoning_content

    finish_reason = choice.get("finish_reason", "unknown")
    return f"Empty assistant content. finish_reason={finish_reason}"
