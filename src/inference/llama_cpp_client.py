from __future__ import annotations

from typing import Any

import httpx


class LlamaCppClient:
    """Small client for llama.cpp OpenAI-compatible chat-completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]
