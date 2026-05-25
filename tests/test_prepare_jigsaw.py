from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.prepare_jigsaw import convert_jigsaw_csv
from src.data.jsonl import validate_jsonl


FIELDNAMES = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def write_jigsaw_csv(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def jigsaw_row(**overrides: str) -> dict[str, str]:
    row = {
        "id": "abc",
        "comment_text": "A public comment",
        "toxic": "0",
        "severe_toxic": "0",
        "obscene": "0",
        "threat": "0",
        "insult": "0",
        "identity_hate": "0",
    }
    row.update(overrides)
    return row


def convert_one(tmp_path: Path, row: dict[str, str]) -> dict:
    input_path = tmp_path / "jigsaw.csv"
    output_path = tmp_path / "normalized.jsonl"
    write_jigsaw_csv(input_path, [row])

    convert_jigsaw_csv(
        input_path=input_path,
        output_path=output_path,
        split="train",
    )

    return validate_jsonl(output_path)[0].model_dump()


def test_no_active_labels_maps_to_sin_riesgo_and_allow(tmp_path: Path) -> None:
    record = convert_one(tmp_path, jigsaw_row())

    assert record["risk_labels"] == ["sin_riesgo"]
    assert record["action"] == "allow"


@pytest.mark.parametrize(
    ("source_label", "expected_risk_label"),
    [
        ("toxic", "insulto_toxicidad"),
        ("insult", "insulto_toxicidad"),
        ("threat", "amenaza_violencia"),
        ("identity_hate", "odio_discriminacion"),
        ("obscene", "sexual_nsfw"),
    ],
)
def test_active_label_maps_to_expected_risk_label(
    tmp_path: Path,
    source_label: str,
    expected_risk_label: str,
) -> None:
    record = convert_one(tmp_path, jigsaw_row(**{source_label: "1"}))

    assert record["risk_labels"] == [expected_risk_label]
    assert record["action"] == "review"


def test_multiple_active_labels_generate_multiple_risk_labels(tmp_path: Path) -> None:
    record = convert_one(
        tmp_path,
        jigsaw_row(toxic="1", threat="1", identity_hate="1", obscene="1"),
    )

    assert record["risk_labels"] == [
        "insulto_toxicidad",
        "odio_discriminacion",
        "amenaza_violencia",
        "sexual_nsfw",
    ]
    assert record["action"] == "review"


def test_generated_jsonl_validates_and_preserves_normalized_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "jigsaw.csv"
    output_path = tmp_path / "normalized.jsonl"
    write_jigsaw_csv(input_path, [jigsaw_row(id="row-1", comment_text="Hello")])

    result = convert_jigsaw_csv(
        input_path=input_path,
        output_path=output_path,
        split="validation",
        source_dataset="custom_jigsaw",
    )
    records = validate_jsonl(output_path)

    assert result.written == 1
    assert result.skipped == 0
    assert records[0].id == "custom_jigsaw_000001"
    assert records[0].source_dataset == "custom_jigsaw"
    assert records[0].text == "Hello"
    assert records[0].topic == "otro"
    assert records[0].split == "validation"
    assert records[0].original_labels == {
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
    }
    assert records[0].metadata == {"original_index": 1, "original_id": "row-1"}


def test_max_examples_limits_valid_written_examples(tmp_path: Path) -> None:
    input_path = tmp_path / "jigsaw.csv"
    output_path = tmp_path / "normalized.jsonl"
    write_jigsaw_csv(input_path, [jigsaw_row(id="1"), jigsaw_row(id="2"), jigsaw_row(id="3")])

    result = convert_jigsaw_csv(
        input_path=input_path,
        output_path=output_path,
        split="test",
        max_examples=2,
    )
    records = validate_jsonl(output_path)

    assert result.written == 2
    assert [record.id for record in records] == ["jigsaw_000001", "jigsaw_000002"]


def test_missing_required_columns_fail_clearly(tmp_path: Path) -> None:
    input_path = tmp_path / "jigsaw.csv"
    output_path = tmp_path / "normalized.jsonl"
    write_jigsaw_csv(
        input_path,
        [{"comment_text": "Hello", "toxic": "0"}],
        fieldnames=["comment_text", "toxic"],
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        convert_jigsaw_csv(input_path=input_path, output_path=output_path, split="train")


def test_invalid_rows_fail_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "jigsaw.csv"
    output_path = tmp_path / "normalized.jsonl"
    write_jigsaw_csv(input_path, [jigsaw_row(toxic="maybe")])

    with pytest.raises(ValueError, match="row 1"):
        convert_jigsaw_csv(input_path=input_path, output_path=output_path, split="train")


def test_skip_invalid_rows_counts_skips_and_keeps_original_index_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "jigsaw.csv"
    output_path = tmp_path / "normalized.jsonl"
    write_jigsaw_csv(
        input_path,
        [
            jigsaw_row(toxic="maybe"),
            jigsaw_row(id="valid-2", threat="1"),
            jigsaw_row(id="valid-3"),
        ],
    )

    result = convert_jigsaw_csv(
        input_path=input_path,
        output_path=output_path,
        split="train",
        max_examples=1,
        skip_invalid_rows=True,
    )
    records = validate_jsonl(output_path)

    assert result.written == 1
    assert result.skipped == 1
    assert records[0].id == "jigsaw_000002"
    assert records[0].metadata["original_index"] == 2
