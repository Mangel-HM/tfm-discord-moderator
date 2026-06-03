from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.jsonl import read_jsonl, validate_jsonl, write_jsonl
from src.domain.schemas import ALLOWED_RISK_LABELS, NormalizedExample


@dataclass(frozen=True)
class SamplingSummary:
    examples_read: int
    examples_written: int
    input_counts: dict[str, int]
    output_counts: dict[str, int]
    selected_labels: list[str]
    underfilled_labels: dict[str, dict[str, int]]


def count_labels(records: Sequence[NormalizedExample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.risk_labels)
    return dict(counts)


def labels_from_args(labels: Sequence[str] | None) -> list[str]:
    if labels is None:
        return list(ALLOWED_RISK_LABELS)
    return list(dict.fromkeys(labels))


def contains_any(record: NormalizedExample, labels: set[str]) -> bool:
    return any(label in labels for label in record.risk_labels)


def can_add_record(
    record: NormalizedExample,
    *,
    selected_counts: Counter[str],
    target_labels: set[str],
    max_per_label: int,
) -> bool:
    for label in record.risk_labels:
        if label in target_labels and selected_counts[label] >= max_per_label:
            return False
    return True


def sample_unique_records(
    records: Sequence[NormalizedExample],
    *,
    max_per_label: int,
    seed: int,
    include_labels: Sequence[str] | None = None,
    exclude_labels: Sequence[str] | None = None,
    shuffle_output: bool = False,
) -> tuple[list[NormalizedExample], list[str], dict[str, dict[str, int]]]:
    if max_per_label < 1:
        raise ValueError("max_per_label must be greater than zero")

    excluded = set(exclude_labels or [])
    selected_labels = [label for label in labels_from_args(include_labels) if label not in excluded]
    target_labels = set(selected_labels)

    eligible_records = [
        record
        for record in records
        if not contains_any(record, excluded) and contains_any(record, target_labels)
    ]

    rng = random.Random(seed)
    candidates_by_label: dict[str, list[NormalizedExample]] = {}
    for label in selected_labels:
        candidates = [record for record in eligible_records if label in record.risk_labels]
        rng.shuffle(candidates)
        candidates_by_label[label] = candidates

    selected_by_id: dict[str, NormalizedExample] = {}
    selected_counts: Counter[str] = Counter()
    made_progress = True
    while made_progress:
        made_progress = False
        for label in selected_labels:
            if selected_counts[label] >= max_per_label:
                continue
            for record in candidates_by_label[label]:
                if record.id in selected_by_id:
                    continue
                if not can_add_record(
                    record,
                    selected_counts=selected_counts,
                    target_labels=target_labels,
                    max_per_label=max_per_label,
                ):
                    continue
                selected_by_id[record.id] = record
                selected_counts.update(
                    risk_label for risk_label in record.risk_labels if risk_label in target_labels
                )
                made_progress = True
                break

    input_order = {record.id: index for index, record in enumerate(records)}
    selected_records = sorted(
        selected_by_id.values(),
        key=lambda record: input_order[record.id],
    )
    if shuffle_output:
        rng.shuffle(selected_records)

    available_counts = count_labels(eligible_records)
    output_counts = count_labels(selected_records)
    underfilled = {
        label: {
            "available": available_counts.get(label, 0),
            "written": output_counts.get(label, 0),
            "requested": max_per_label,
        }
        for label in selected_labels
        if output_counts.get(label, 0) < max_per_label
    }

    return selected_records, selected_labels, underfilled


def sample_normalized_dataset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    max_per_label: int,
    seed: int = 42,
    include_labels: Sequence[str] | None = None,
    exclude_labels: Sequence[str] | None = None,
    shuffle_output: bool = False,
) -> SamplingSummary:
    records = read_jsonl(input_path)
    selected_records, selected_labels, underfilled = sample_unique_records(
        records,
        max_per_label=max_per_label,
        seed=seed,
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        shuffle_output=shuffle_output,
    )

    validated_selected_records = [
        NormalizedExample.model_validate(record.model_dump()) for record in selected_records
    ]
    write_jsonl(output_path, validated_selected_records)
    validated_records = validate_jsonl(output_path)

    return SamplingSummary(
        examples_read=len(records),
        examples_written=len(validated_records),
        input_counts=count_labels(records),
        output_counts=count_labels(validated_records),
        selected_labels=selected_labels,
        underfilled_labels=underfilled,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a normalized JSONL dataset by risk label.")
    parser.add_argument("--input", required=True, help="Input normalized JSONL path.")
    parser.add_argument("--output", required=True, help="Output normalized JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Defaults to 42.")
    parser.add_argument(
        "--max-per-label",
        type=int,
        required=True,
        help="Maximum number of final examples per included risk label.",
    )
    parser.add_argument(
        "--include-label",
        action="append",
        choices=ALLOWED_RISK_LABELS,
        default=None,
        help="Risk label to include. Can be repeated. Defaults to all labels.",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        choices=ALLOWED_RISK_LABELS,
        default=None,
        help="Risk label to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle selected records before writing the output JSONL.",
    )
    return parser


def print_counts(title: str, counts: dict[str, int]) -> None:
    print(title)
    for label in ALLOWED_RISK_LABELS:
        print(f"  {label}: {counts.get(label, 0)}")


def main(argv: Sequence[str] | None = None) -> SamplingSummary:
    args = build_parser().parse_args(argv)
    summary = sample_normalized_dataset(
        input_path=args.input,
        output_path=args.output,
        max_per_label=args.max_per_label,
        seed=args.seed,
        include_labels=args.include_label,
        exclude_labels=args.exclude_label,
        shuffle_output=args.shuffle_output,
    )
    print(f"Examples read: {summary.examples_read}")
    print(f"Examples written: {summary.examples_written}")
    print_counts("Input risk_label counts:", summary.input_counts)
    print_counts("Output risk_label counts:", summary.output_counts)
    if summary.underfilled_labels:
        print("Underfilled labels:")
        for label, data in summary.underfilled_labels.items():
            print(
                f"  {label}: wrote {data['written']} of {data['requested']} "
                f"(available {data['available']})"
            )
    else:
        print("Underfilled labels: none")
    return summary


if __name__ == "__main__":
    main()
