from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts import classify_sample


class FakeClient:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        assert '"topic"' in messages[1]["content"]
        assert '"risk_labels"' in messages[1]["content"]
        return (
            '{"topic":"otro","risk_labels":["sin_riesgo"],"action":"allow",'
            '"confidence":0.9,"rationale":"No moderation risk."}'
        )


def test_classify_sample_outputs_normalized_json(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    sample_path = tmp_path / "messages_sample.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "message_id": "discord-1",
                "channel": "general",
                "author_role": "member",
                "context": ["user_a: hello"],
                "text": "Hello everyone.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        llm_base_url="http://127.0.0.1:8001/v1",
        llm_api_key="not-needed",
        llm_model="discord-qwen-local",
    )
    monkeypatch.setattr(classify_sample, "get_settings", lambda: settings)
    monkeypatch.setattr(classify_sample, "LlamaCppClient", FakeClient)
    monkeypatch.setattr(classify_sample, "SAMPLE_PATH", sample_path)

    asyncio.run(classify_sample.main())

    output = json.loads(capsys.readouterr().out)
    assert output["message_id"] == "discord-1"
    assert output["topic"] == "otro"
    assert output["risk_labels"] == ["sin_riesgo"]
    assert output["action"] == "allow"
    assert "label" not in output
    assert "risk" not in output
