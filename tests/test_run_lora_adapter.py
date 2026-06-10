from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_lora_adapter import (
    build_lora_messages,
    build_parser,
    run_lora_adapter,
)
from src.classification.baseline_classifier import parse_normalized_classification
from src.data.jsonl import write_jsonl
from src.domain.schemas import (
    ALLOWED_ACTIONS,
    ALLOWED_RISK_LABELS,
    ALLOWED_TOPICS,
    NormalizedExample,
)


class FakeGenerator:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def generate(self, example: NormalizedExample) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def make_example(example_id: str = "jigsaw_000001") -> NormalizedExample:
    return NormalizedExample.model_validate(
        {
            "id": example_id,
            "source_dataset": "jigsaw",
            "text": "You are awful.",
            "topic": "otro",
            "risk_labels": ["insulto_toxicidad"],
            "action": "review",
            "split": "test",
            "original_labels": {"toxic": 1},
            "metadata": {},
        }
    )


def test_lora_prompt_matches_sft_taxonomy_and_excludes_removed_label() -> None:
    messages = build_lora_messages(make_example())
    prompt = messages[1]["content"]

    assert "conservative Discord moderation classifier" in messages[0]["content"]
    for topic in ALLOWED_TOPICS:
        assert topic in prompt
    for risk_label in ALLOWED_RISK_LABELS:
        assert risk_label in prompt
    for action in ALLOWED_ACTIONS:
        assert action in prompt
    assert "spam_fraude" not in prompt
    assert "You are awful." in prompt


def test_parse_lora_clean_json_without_confidence() -> None:
    result = parse_normalized_classification(
        '{"topic":"otro","risk_labels":["sin_riesgo"],"action":"allow"}'
    )

    assert result.topic == "otro"
    assert result.risk_labels == ["sin_riesgo"]
    assert result.action == "allow"
    assert result.confidence is None


def test_parse_lora_json_inside_text() -> None:
    result = parse_normalized_classification(
        'Result:\n{"topic":"otro","risk_labels":["insulto_toxicidad"],"action":"review"}'
    )

    assert result.risk_labels == ["insulto_toxicidad"]
    assert result.action == "review"


def test_parse_lora_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="did not return JSON"):
        parse_normalized_classification("not json")


def test_lora_parser_rejects_incompatible_precision_flags() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "input.jsonl",
                "--output",
                "output.jsonl",
                "--adapter-dir",
                "outputs/lora",
                "--model-name-or-path",
                "Qwen/Qwen3.5-2B",
                "--bf16",
                "--fp16",
            ]
        )


def test_lora_parser_rejects_invalid_numeric_limits() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "input.jsonl",
                "--output",
                "output.jsonl",
                "--adapter-dir",
                "outputs/lora",
                "--model-name-or-path",
                "Qwen/Qwen3.5-2B",
                "--max-examples",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "input.jsonl",
                "--output",
                "output.jsonl",
                "--adapter-dir",
                "outputs/lora",
                "--model-name-or-path",
                "Qwen/Qwen3.5-2B",
                "--temperature",
                "-0.1",
            ]
        )


def test_run_lora_adapter_writes_evaluator_compatible_predictions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_example()])

    summary = run_lora_adapter(
        input_path=input_path,
        output_path=output_path,
        generator=FakeGenerator(
            ['{"topic":"otro","risk_labels":["insulto_toxicidad"],"action":"review"}']
        ),
    )

    prediction = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert summary.processed == 1
    assert summary.parsed == 1
    assert summary.parse_errors == 0
    assert prediction["gold_action"] == "review"
    assert prediction["pred_action"] == "review"
    assert prediction["pred_risk_labels"] == ["insulto_toxicidad"]
    assert prediction["confidence"] is None
    assert prediction["rationale"] is None
    assert prediction["parse_error"] is None


def test_run_lora_adapter_can_continue_on_parse_error(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_example()])

    summary = run_lora_adapter(
        input_path=input_path,
        output_path=output_path,
        generator=FakeGenerator(["not json"]),
        continue_on_error=True,
    )

    prediction = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert summary.parsed == 0
    assert summary.parse_errors == 1
    assert prediction["parse_error"]
