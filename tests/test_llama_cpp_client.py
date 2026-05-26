from __future__ import annotations

import asyncio
from typing import Any

from src.inference.llama_cpp_client import LlamaCppClient


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeAsyncClient:
    posted_payloads: list[dict[str, Any]] = []
    response_payload: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"topic":"otro"}'},
            }
        ]
    }

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:
        self.posted_payloads.append(json)
        return FakeResponse(self.response_payload)


def test_chat_disables_qwen_thinking_by_default(monkeypatch) -> None:
    FakeAsyncClient.posted_payloads = []
    monkeypatch.setattr("src.inference.llama_cpp_client.httpx.AsyncClient", FakeAsyncClient)
    client = LlamaCppClient()

    asyncio.run(client.chat([{"role": "user", "content": "Classify this."}]))

    payload = FakeAsyncClient.posted_payloads[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_can_omit_thinking_setting(monkeypatch) -> None:
    FakeAsyncClient.posted_payloads = []
    monkeypatch.setattr("src.inference.llama_cpp_client.httpx.AsyncClient", FakeAsyncClient)
    client = LlamaCppClient(enable_thinking=None)

    asyncio.run(client.chat([{"role": "user", "content": "Classify this."}]))

    payload = FakeAsyncClient.posted_payloads[0]
    assert "chat_template_kwargs" not in payload


def test_chat_returns_reasoning_content_when_content_is_empty(monkeypatch) -> None:
    monkeypatch.setattr("src.inference.llama_cpp_client.httpx.AsyncClient", FakeAsyncClient)
    FakeAsyncClient.response_payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "reasoning_content": "Thinking Process: no final answer was produced.",
                },
            }
        ]
    }
    client = LlamaCppClient()

    output = asyncio.run(client.chat([{"role": "user", "content": "Classify this."}]))

    assert output == "Thinking Process: no final answer was produced."
