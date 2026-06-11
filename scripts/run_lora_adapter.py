from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from scripts.build_sft_dataset import SFT_SYSTEM_PROMPT, build_sft_user_prompt
from scripts.train_lora import positive_int
from src.classification.baseline_classifier import parse_normalized_classification
from src.classification.lora_classifier import get_torch_dtype, load_tokenizer_with_chat_template
from src.data.jsonl import read_jsonl
from src.domain.schemas import BaselinePrediction, NormalizedExample


class LoraGenerator(Protocol):
    def generate(self, example: NormalizedExample) -> str:
        pass


@dataclass(frozen=True)
class RunSummary:
    processed: int
    parsed: int
    parse_errors: int
    mean_latency_ms: float


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_lora_messages(example: NormalizedExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": build_sft_user_prompt(example)},
    ]


def build_prediction_from_lora_output(
    example: NormalizedExample,
    *,
    raw_output: str,
    latency_ms: float,
    continue_on_error: bool,
) -> BaselinePrediction:
    try:
        result = parse_normalized_classification(raw_output)
    except ValueError as exc:
        if not continue_on_error:
            raise
        return BaselinePrediction.from_parse_error(
            example,
            latency_ms=latency_ms,
            raw_response=raw_output,
            parse_error=str(exc),
        )
    return BaselinePrediction.from_example(
        example,
        pred_topic=result.topic,
        pred_risk_labels=result.risk_labels,
        pred_action=result.action,
        confidence=result.confidence,
        rationale=result.rationale,
        latency_ms=latency_ms,
        raw_response=raw_output,
    )


def run_lora_adapter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    generator: LoraGenerator,
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
            started = perf_counter()
            try:
                raw_output = generator.generate(example)
            except Exception as exc:
                latency_ms = (perf_counter() - started) * 1000
                if not continue_on_error:
                    raise
                prediction = BaselinePrediction.from_parse_error(
                    example,
                    latency_ms=latency_ms,
                    raw_response="",
                    parse_error=f"Inference error: {exc}",
                )
            else:
                latency_ms = (perf_counter() - started) * 1000
                prediction = build_prediction_from_lora_output(
                    example,
                    raw_output=raw_output,
                    latency_ms=latency_ms,
                    continue_on_error=continue_on_error,
                )
            predictions.append(prediction)
            file.write(json.dumps(prediction.model_dump(), ensure_ascii=False, sort_keys=True))
            file.write("\n")

    processed = len(predictions)
    parse_errors = sum(1 for prediction in predictions if prediction.parse_error)
    total_latency = sum(prediction.latency_ms for prediction in predictions)
    return RunSummary(
        processed=processed,
        parsed=processed - parse_errors,
        parse_errors=parse_errors,
        mean_latency_ms=(total_latency / processed if processed else 0.0),
    )


def load_tokenizer(*, adapter_dir: str, model_name_or_path: str):
    return load_tokenizer_with_chat_template(
        adapter_dir=adapter_dir,
        model_name_or_path=model_name_or_path,
    )


class TransformersLoraGenerator:
    def __init__(
        self,
        *,
        adapter_dir: str,
        model_name_or_path: str,
        max_new_tokens: int,
        temperature: float,
        bf16: bool,
        fp16: bool,
        seed: int,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, set_seed

        random.seed(seed)
        set_seed(seed)
        torch.manual_seed(seed)

        self.tokenizer = load_tokenizer(
            adapter_dir=adapter_dir,
            model_name_or_path=model_name_or_path,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=get_torch_dtype(bf16=bf16, fp16=fp16),
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, example: NormalizedExample) -> str:
        import torch

        messages = build_lora_messages(example)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a base Transformers model plus LoRA adapter over normalized examples."
    )
    parser.add_argument("--input", required=True, help="Input normalized JSONL path.")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path.")
    parser.add_argument("--adapter-dir", required=True, help="Directory containing LoRA adapter.")
    parser.add_argument("--model-name-or-path", required=True, help="Base Transformers model.")
    parser.add_argument("--max-examples", type=positive_int, default=None)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--temperature", type=non_negative_float, default=0.0)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write parse/inference errors as prediction rows instead of stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> RunSummary:
    args = build_parser().parse_args(argv)
    generator = TransformersLoraGenerator(
        adapter_dir=args.adapter_dir,
        model_name_or_path=args.model_name_or_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
    )
    summary = run_lora_adapter(
        input_path=args.input,
        output_path=args.output,
        generator=generator,
        max_examples=args.max_examples,
        continue_on_error=args.continue_on_error,
    )
    print(f"Processed examples: {summary.processed}")
    print(f"Parsed predictions: {summary.parsed}")
    print(f"Parse errors: {summary.parse_errors}")
    print(f"Mean latency ms: {summary.mean_latency_ms:.1f}")
    return summary


if __name__ == "__main__":
    main()
