from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.jsonl import read_jsonl
from src.domain.schemas import (
    ALLOWED_ACTIONS,
    ALLOWED_RISK_LABELS,
    ALLOWED_TOPICS,
    NormalizedExample,
)


SFT_SYSTEM_PROMPT = """You are a conservative Discord moderation classifier.
Classify English messages only. Return only valid JSON and no markdown.
Use only the allowed topics, risk labels, and actions listed by the user."""


@dataclass(frozen=True)
class BuildSftSummary:
    examples_read: int
    examples_written: int
    examples_excluded: int
    excluded_ids: int


def build_sft_user_prompt(example: NormalizedExample) -> str:
    topics = ", ".join(ALLOWED_TOPICS)
    risk_labels = ", ".join(ALLOWED_RISK_LABELS)
    actions = ", ".join(ALLOWED_ACTIONS)

    return f"""
Classify this English message for Discord moderation.

Allowed topics: [{topics}]
Allowed risk_labels: [{risk_labels}]
Allowed actions: [{actions}]

Return only valid JSON with exactly these fields:
- "topic"
- "risk_labels"
- "action"

Rules:
- If there is no moderation risk, use risk_labels = ["sin_riesgo"] and action = "allow".
- Do not combine sin_riesgo with any other risk label.
- Do not invent labels outside the allowed lists.
- Prefer "review" for risky content unless the message clearly suggests a warning or deletion candidate.

Message:
{example.text}
""".strip()


def build_assistant_target(
    example: NormalizedExample,
    *,
    include_rationale: bool = False,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "topic": example.topic,
        "risk_labels": example.risk_labels,
        "action": example.action,
    }
    if include_rationale:
        target["rationale"] = (
            f"Gold labels indicate topic '{example.topic}', risk_labels "
            f"{example.risk_labels}, and action '{example.action}'."
        )
    return target


def build_sft_record(
    example: NormalizedExample,
    *,
    include_rationale: bool = False,
) -> dict[str, Any]:
    assistant_content = json.dumps(
        build_assistant_target(example, include_rationale=include_rationale),
        ensure_ascii=False,
        sort_keys=True,
    )
    json.loads(assistant_content)

    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": build_sft_user_prompt(example)},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "id": example.id,
            "source_dataset": example.source_dataset,
            "split": example.split,
            "topic": example.topic,
            "risk_labels": example.risk_labels,
        },
    }


def read_excluded_ids(paths: Sequence[str | Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                record_id = payload.get("id") if isinstance(payload, dict) else None
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError(f"Missing string id at {path}:{line_number}")
                excluded.add(record_id)
    return excluded


def write_sft_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            assistant_content = record["messages"][2]["content"]
            json.loads(assistant_content)
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def build_sft_dataset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    exclude_ids_from: Sequence[str | Path] = (),
    max_examples: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
    include_rationale: bool = False,
) -> BuildSftSummary:
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be greater than zero")

    examples = read_jsonl(input_path)
    excluded_ids = read_excluded_ids(exclude_ids_from)
    selected = [example for example in examples if example.id not in excluded_ids]
    examples_excluded = len(examples) - len(selected)

    if shuffle:
        random.Random(seed).shuffle(selected)
    if max_examples is not None:
        selected = selected[:max_examples]

    records = [
        build_sft_record(example, include_rationale=include_rationale) for example in selected
    ]
    write_sft_jsonl(output_path, records)
    return BuildSftSummary(
        examples_read=len(examples),
        examples_written=len(records),
        examples_excluded=examples_excluded,
        excluded_ids=len(excluded_ids),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a chat/SFT JSONL dataset from normalized examples."
    )
    parser.add_argument("--input", required=True, help="Input normalized JSONL path.")
    parser.add_argument("--output", required=True, help="Output chat/SFT JSONL path.")
    parser.add_argument(
        "--exclude-ids-from",
        action="append",
        default=[],
        help="JSONL path whose record ids should be excluded. Can be repeated.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of final examples to write.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Defaults to 42.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle examples before writing.")
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        help="Include a brief synthetic rationale in the assistant target.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> BuildSftSummary:
    args = build_parser().parse_args(argv)
    summary = build_sft_dataset(
        input_path=args.input,
        output_path=args.output,
        exclude_ids_from=args.exclude_ids_from,
        max_examples=args.max_examples,
        seed=args.seed,
        shuffle=args.shuffle,
        include_rationale=args.include_rationale,
    )
    print(f"Examples read: {summary.examples_read}")
    print(f"Excluded ids loaded: {summary.excluded_ids}")
    print(f"Examples excluded: {summary.examples_excluded}")
    print(f"Examples written: {summary.examples_written}")
    return summary


if __name__ == "__main__":
    main()
