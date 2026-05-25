from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.data.jsonl import write_jsonl
from src.domain.schemas import ALLOWED_SPLITS, NormalizedExample


JIGSAW_LABEL_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
REQUIRED_COLUMNS = ("comment_text", *JIGSAW_LABEL_COLUMNS)
RISK_LABEL_ORDER = (
    "insulto_toxicidad",
    "odio_discriminacion",
    "amenaza_violencia",
    "sexual_nsfw",
)


@dataclass(frozen=True)
class ConversionResult:
    written: int
    skipped: int


def parse_label_value(value: str | None, *, row_number: int, column: str) -> int:
    if value is None:
        raise ValueError(f"row {row_number}: missing value for {column}")
    normalized = value.strip()
    if normalized in {"0", "0.0"}:
        return 0
    if normalized in {"1", "1.0"}:
        return 1
    raise ValueError(f"row {row_number}: invalid value for {column}: {value!r}")


def map_risk_labels(original_labels: dict[str, int]) -> list[str]:
    labels: list[str] = []
    if original_labels["toxic"] or original_labels["severe_toxic"] or original_labels["insult"]:
        labels.append("insulto_toxicidad")
    if original_labels["identity_hate"]:
        labels.append("odio_discriminacion")
    if original_labels["threat"]:
        labels.append("amenaza_violencia")
    if original_labels["obscene"]:
        labels.append("sexual_nsfw")
    return [label for label in RISK_LABEL_ORDER if label in labels] or ["sin_riesgo"]


def row_to_example(
    row: dict[str, str],
    *,
    row_number: int,
    split: str,
    source_dataset: str,
) -> NormalizedExample:
    text = row.get("comment_text", "")
    original_labels = {
        column: parse_label_value(row.get(column), row_number=row_number, column=column)
        for column in JIGSAW_LABEL_COLUMNS
    }
    risk_labels = map_risk_labels(original_labels)
    metadata: dict[str, Any] = {"original_index": row_number}
    original_id = row.get("id", "").strip()
    if original_id:
        metadata["original_id"] = original_id

    return NormalizedExample.model_validate(
        {
            "id": f"{source_dataset}_{row_number:06d}",
            "source_dataset": source_dataset,
            "text": text,
            "topic": "otro",
            "risk_labels": risk_labels,
            "action": "allow" if risk_labels == ["sin_riesgo"] else "review",
            "split": split,
            "original_labels": original_labels,
            "metadata": metadata,
        }
    )


def ensure_required_columns(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("Input CSV does not contain a header")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def convert_jigsaw_csv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    split: str,
    max_examples: int | None = None,
    source_dataset: str = "jigsaw",
    skip_invalid_rows: bool = False,
) -> ConversionResult:
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be greater than zero")

    records: list[NormalizedExample] = []
    skipped = 0
    with Path(input_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        ensure_required_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=1):
            try:
                records.append(
                    row_to_example(
                        row,
                        row_number=row_number,
                        split=split,
                        source_dataset=source_dataset,
                    )
                )
            except (ValueError, ValidationError) as exc:
                if not skip_invalid_rows:
                    raise ValueError(f"Could not convert row {row_number}: {exc}") from exc
                skipped += 1
                continue
            if max_examples is not None and len(records) >= max_examples:
                break

    write_jsonl(output_path, records)
    return ConversionResult(written=len(records), skipped=skipped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Jigsaw CSV data to normalized JSONL.")
    parser.add_argument("--input", required=True, help="Input Jigsaw CSV path.")
    parser.add_argument("--output", required=True, help="Output normalized JSONL path.")
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS, help="Dataset split.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of valid examples to write.",
    )
    parser.add_argument(
        "--source-dataset",
        default="jigsaw",
        help='Source dataset name stored in each record. Defaults to "jigsaw".',
    )
    parser.add_argument(
        "--skip-invalid-rows",
        action="store_true",
        help="Skip invalid rows and report how many were omitted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = convert_jigsaw_csv(
        input_path=args.input,
        output_path=args.output,
        split=args.split,
        max_examples=args.max_examples,
        source_dataset=args.source_dataset,
        skip_invalid_rows=args.skip_invalid_rows,
    )
    print(f"Wrote {result.written} examples to {args.output}")
    if result.skipped:
        print(f"Skipped {result.skipped} invalid rows")


if __name__ == "__main__":
    main()
