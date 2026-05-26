from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from src.classification.baseline_classifier import build_prediction_from_raw_output
from src.classification.prompts import BASELINE_SYSTEM_PROMPT, build_baseline_prompt
from src.data.jsonl import read_jsonl
from src.domain.schemas import BaselinePrediction, NormalizedExample
from src.inference.llama_cpp_client import LlamaCppClient


class ChatClient(Protocol):
    async def chat(self, messages: list[dict[str, str]]) -> str:
        pass


@dataclass(frozen=True)
class RunSummary:
    processed: int
    parsed: int
    parse_errors: int
    mean_latency_ms: float


async def classify_one(
    example: NormalizedExample,
    *,
    client: ChatClient,
    continue_on_error: bool,
) -> BaselinePrediction:
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": build_baseline_prompt(example)},
    ]
    started = perf_counter()
    try:
        raw_output = await client.chat(messages)
    except Exception as exc:
        latency_ms = (perf_counter() - started) * 1000
        if not continue_on_error:
            raise
        return BaselinePrediction.from_parse_error(
            example,
            latency_ms=latency_ms,
            raw_response="",
            parse_error=f"Inference error: {exc}",
        )
    latency_ms = (perf_counter() - started) * 1000
    return build_prediction_from_raw_output(
        example,
        raw_output=raw_output,
        latency_ms=latency_ms,
        continue_on_error=continue_on_error,
    )


async def run_baseline(
    *,
    input_path: str | Path,
    output_path: str | Path,
    client: ChatClient,
    max_examples: int | None = None,
    continue_on_error: bool = False,
) -> RunSummary:
    examples = read_jsonl(input_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[BaselinePrediction] = []
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            prediction = await classify_one(
                example,
                client=client,
                continue_on_error=continue_on_error,
            )
            predictions.append(prediction)
            file.write(json.dumps(prediction.model_dump(), ensure_ascii=False, sort_keys=True))
            file.write("\n")

    processed = len(predictions)
    parse_errors = sum(1 for prediction in predictions if prediction.parse_error)
    parsed = processed - parse_errors
    total_latency = sum(prediction.latency_ms for prediction in predictions)
    return RunSummary(
        processed=processed,
        parsed=parsed,
        parse_errors=parse_errors,
        mean_latency_ms=(total_latency / processed if processed else 0.0),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the normalized llama.cpp baseline.")
    parser.add_argument("--input", required=True, help="Input normalized JSONL path.")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to classify.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible llama.cpp base URL.",
    )
    parser.add_argument("--model", default="discord-qwen-local", help="Model name or alias.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--api-key",
        default="not-needed",
        help="Dummy API key value for OpenAI-compatible clients that require one.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write parse/inference errors as prediction rows instead of stopping.",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> RunSummary:
    args = build_parser().parse_args(argv)
    client = LlamaCppClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout,
        temperature=0.0,
        max_tokens=256,
    )
    summary = await run_baseline(
        input_path=args.input,
        output_path=args.output,
        client=client,
        max_examples=args.max_examples,
        continue_on_error=args.continue_on_error,
    )
    print(f"Processed examples: {summary.processed}")
    print(f"Parsed predictions: {summary.parsed}")
    print(f"Parse errors: {summary.parse_errors}")
    print(f"Mean latency ms: {summary.mean_latency_ms:.1f}")
    return summary


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
