from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.domain.schemas import NormalizedExample


NormalizedRecord = NormalizedExample | Mapping[str, Any]


def read_jsonl(path: str | Path) -> list[NormalizedExample]:
    records: list[NormalizedExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
            try:
                records.append(NormalizedExample.model_validate(payload))
            except ValidationError as exc:
                raise ValueError(f"Invalid normalized example at line {line_number}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[NormalizedRecord]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            example = NormalizedExample.model_validate(record)
            payload = example.model_dump()
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def validate_jsonl(path: str | Path) -> list[NormalizedExample]:
    return read_jsonl(path)
