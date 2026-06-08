from __future__ import annotations

import json
from pathlib import Path

from scripts.build_sft_dataset import build_sft_dataset, build_sft_record
from src.data.jsonl import write_jsonl
from src.domain.schemas import NormalizedExample


def make_example(
    example_id: str,
    *,
    text: str | None = None,
    risk_labels: list[str] | None = None,
) -> NormalizedExample:
    labels = risk_labels or ["sin_riesgo"]
    return NormalizedExample.model_validate(
        {
            "id": example_id,
            "source_dataset": "jigsaw",
            "text": text or f"Message {example_id}",
            "topic": "otro",
            "risk_labels": labels,
            "action": "allow" if labels == ["sin_riesgo"] else "review",
            "split": "train",
            "original_labels": {"source_id": example_id},
            "metadata": {"language": "en"},
        }
    )


def read_jsonl_payloads(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assistant_payload(record: dict) -> dict:
    return json.loads(record["messages"][2]["content"])


def test_build_sft_record_from_normalized_example() -> None:
    example = make_example(
        "jigsaw_000001",
        text="Please stop posting the same link everywhere.",
        risk_labels=["spam_fraude"],
    )

    record = build_sft_record(example)

    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "user"
    assert record["messages"][2]["role"] == "assistant"
    assert "Please stop posting the same link everywhere." in record["messages"][1]["content"]
    assert "spam_fraude" in record["messages"][1]["content"]
    assert assistant_payload(record) == {
        "action": "review",
        "risk_labels": ["spam_fraude"],
        "topic": "otro",
    }
    assert record["metadata"] == {
        "id": "jigsaw_000001",
        "source_dataset": "jigsaw",
        "split": "train",
        "topic": "otro",
        "risk_labels": ["spam_fraude"],
    }


def test_assistant_content_is_valid_json_without_confidence() -> None:
    record = build_sft_record(make_example("jigsaw_000001"))

    payload = assistant_payload(record)

    assert payload["topic"] == "otro"
    assert "confidence" not in payload


def test_include_rationale_adds_synthetic_rationale() -> None:
    record = build_sft_record(make_example("jigsaw_000001"), include_rationale=True)

    payload = assistant_payload(record)

    assert "rationale" in payload
    assert "Gold labels indicate" in payload["rationale"]


def test_build_sft_dataset_excludes_ids_from_jsonl_files(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    exclude_path = tmp_path / "exclude.jsonl"
    output_path = tmp_path / "sft.jsonl"
    write_jsonl(input_path, [make_example("keep-1"), make_example("drop-1")])
    exclude_path.write_text('{"id":"drop-1","parse_error":null}\n', encoding="utf-8")

    summary = build_sft_dataset(
        input_path=input_path,
        output_path=output_path,
        exclude_ids_from=[exclude_path],
    )

    records = read_jsonl_payloads(output_path)
    assert summary.examples_read == 2
    assert summary.examples_written == 1
    assert summary.examples_excluded == 1
    assert [record["metadata"]["id"] for record in records] == ["keep-1"]


def test_shuffle_is_reproducible_with_seed(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    examples = [make_example(f"example-{index}") for index in range(10)]
    write_jsonl(input_path, examples)

    build_sft_dataset(input_path=input_path, output_path=first_output, seed=7, shuffle=True)
    build_sft_dataset(input_path=input_path, output_path=second_output, seed=7, shuffle=True)

    first_ids = [record["metadata"]["id"] for record in read_jsonl_payloads(first_output)]
    second_ids = [record["metadata"]["id"] for record in read_jsonl_payloads(second_output)]
    assert first_ids == second_ids
    assert first_ids != [example.id for example in examples]


def test_max_examples_limits_written_records(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "sft.jsonl"
    write_jsonl(input_path, [make_example(f"example-{index}") for index in range(5)])

    summary = build_sft_dataset(input_path=input_path, output_path=output_path, max_examples=2)

    records = read_jsonl_payloads(output_path)
    assert summary.examples_written == 2
    assert [record["metadata"]["id"] for record in records] == ["example-0", "example-1"]
