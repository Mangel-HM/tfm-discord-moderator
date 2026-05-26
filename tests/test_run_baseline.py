from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scripts.run_baseline import run_baseline
from src.data.jsonl import write_jsonl
from src.domain.schemas import NormalizedExample


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def chat(self, messages: list[dict[str, str]]) -> str:
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        response = self.responses[self.calls]
        self.calls += 1
        return response


def make_example(example_id: str = "jigsaw_000001") -> NormalizedExample:
    return NormalizedExample.model_validate(
        {
            "id": example_id,
            "source_dataset": "jigsaw",
            "text": "A public comment.",
            "topic": "otro",
            "risk_labels": ["sin_riesgo"],
            "action": "allow",
            "split": "train",
            "original_labels": {"toxic": 0},
            "metadata": {},
        }
    )


def test_run_baseline_writes_predictions_and_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_example()])
    client = FakeClient(
        [
            (
                '{"topic":"otro","risk_labels":["sin_riesgo"],"action":"allow",'
                '"confidence":0.95,"rationale":"No risk."}'
            )
        ]
    )

    summary = asyncio.run(
        run_baseline(
            input_path=input_path,
            output_path=output_path,
            client=client,
            continue_on_error=False,
        )
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    prediction = json.loads(lines[0])
    assert summary.processed == 1
    assert summary.parsed == 1
    assert summary.parse_errors == 0
    assert summary.mean_latency_ms >= 0
    assert prediction["id"] == "jigsaw_000001"
    assert prediction["pred_action"] == "allow"
    assert prediction["parse_error"] is None


def test_run_baseline_can_continue_on_parse_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_example()])
    client = FakeClient(["not json"])

    summary = asyncio.run(
        run_baseline(
            input_path=input_path,
            output_path=output_path,
            client=client,
            continue_on_error=True,
        )
    )

    prediction = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert summary.processed == 1
    assert summary.parsed == 0
    assert summary.parse_errors == 1
    assert prediction["parse_error"]
