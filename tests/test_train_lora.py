from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_lora import (
    REQUIRED_SUMMARY_KEYS,
    build_parser,
    build_training_summary,
    maybe_limit_dataset,
    read_sft_jsonl,
)


def write_sft_record(path: Path, *, message: str = "Please stop insulting people.") -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "Classify messages."},
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "topic": "otro",
                        "risk_labels": ["insulto_toxicidad"],
                        "action": "review",
                    }
                ),
            },
        ],
        "metadata": {"id": "example-1"},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_read_sft_jsonl_reads_valid_chat_records(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    write_sft_record(train_file)

    dataset = read_sft_jsonl(train_file)

    assert len(dataset) == 1
    assert dataset[0]["messages"][1]["content"] == "Please stop insulting people."
    assert dataset[0]["metadata"] == {"id": "example-1"}


def test_read_sft_jsonl_rejects_invalid_assistant_json(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    payload = {
        "messages": [
            {"role": "system", "content": "Classify messages."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "not json"},
        ],
        "metadata": {},
    }
    train_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="assistant content must be valid JSON"):
        read_sft_jsonl(train_file)


def test_maybe_limit_dataset_limits_records(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    records = []
    for index in range(3):
        payload = {
            "messages": [
                {"role": "system", "content": "Classify messages."},
                {"role": "user", "content": f"Message {index}"},
                {"role": "assistant", "content": '{"topic":"otro"}'},
            ],
            "metadata": {"id": f"example-{index}"},
        }
        records.append(json.dumps(payload))
    train_file.write_text("\n".join(records) + "\n", encoding="utf-8")
    dataset = read_sft_jsonl(train_file)

    limited = maybe_limit_dataset(dataset, 2)

    assert len(limited) == 2
    assert limited[1]["metadata"]["id"] == "example-1"


def test_build_training_summary_contains_required_keys(tmp_path: Path) -> None:
    summary = build_training_summary(
        model_name_or_path="test-model",
        train_file=tmp_path / "train.jsonl",
        output_dir=tmp_path / "adapter",
        max_examples=100,
        max_steps=20,
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=1024,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        examples_used=12,
        parameter_counts={"trainable_parameters": 10, "total_parameters": 100},
        final_train_loss=0.42,
        status="completed",
    )

    assert REQUIRED_SUMMARY_KEYS <= summary.keys()
    assert summary["trainable_percentage"] == pytest.approx(10.0)
    assert summary["status"] == "completed"


def test_train_lora_parser_rejects_incompatible_precision_flags() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--train-file",
                "train.jsonl",
                "--output-dir",
                "outputs/lora",
                "--model-name-or-path",
                "test-model",
                "--bf16",
                "--fp16",
            ]
        )


def test_train_lora_parser_rejects_non_positive_limits() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--train-file",
                "train.jsonl",
                "--output-dir",
                "outputs/lora",
                "--model-name-or-path",
                "test-model",
                "--max-steps",
                "0",
            ]
        )
