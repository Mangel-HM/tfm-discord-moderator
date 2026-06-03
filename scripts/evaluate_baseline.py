from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.domain.schemas import BaselinePrediction
from src.evaluation.metrics import evaluate_baseline_predictions


def read_baseline_predictions(path: str | Path) -> list[BaselinePrediction]:
    predictions: list[BaselinePrediction] = []
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
                predictions.append(BaselinePrediction.model_validate(payload))
            except ValidationError as exc:
                raise ValueError(f"Invalid baseline prediction at line {line_number}") from exc
    return predictions


def write_metrics(path: str | Path, metrics: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_baseline_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    ignore_topic: bool = False,
) -> dict:
    predictions = read_baseline_predictions(input_path)
    metrics = evaluate_baseline_predictions(predictions, ignore_topic=ignore_topic)
    write_metrics(output_path, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate normalized baseline JSONL predictions.")
    parser.add_argument("--input", required=True, help="Input baseline predictions JSONL path.")
    parser.add_argument("--output", required=True, help="Output metrics JSON path.")
    parser.add_argument(
        "--ignore-topic",
        action="store_true",
        help='Ignore topic accuracy, useful for Jigsaw where topic="otro" is artificial.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    metrics = evaluate_baseline_file(
        input_path=args.input,
        output_path=args.output,
        ignore_topic=args.ignore_topic,
    )
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Parsed predictions: {metrics['parsed_predictions']}")
    print(f"Parse errors: {metrics['parse_errors']}")
    print(f"Valid JSON rate: {metrics['valid_json_rate']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    return metrics


if __name__ == "__main__":
    main()
