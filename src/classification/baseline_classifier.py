from __future__ import annotations

import json

from pydantic import ValidationError

from src.classification.prompts import (
    BASELINE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_baseline_prompt,
    build_classification_prompt,
)
from src.domain.schemas import (
    BaselineClassification,
    BaselinePrediction,
    ClassificationResult,
    DiscordMessage,
    NormalizedExample,
)
from src.inference.llama_cpp_client import LlamaCppClient


class BaselineClassifier:
    """Prompting baseline: no fine-tuning, only structured instructions."""

    def __init__(self, client: LlamaCppClient, taxonomy: dict):
        self.client = client
        self.taxonomy = taxonomy

    async def classify(self, message: DiscordMessage) -> ClassificationResult:
        user_prompt = build_classification_prompt(message, self.taxonomy)
        raw_output = await self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return parse_classification_result(raw_output)

    async def classify_example(
        self, example: NormalizedExample
    ) -> tuple[BaselineClassification, str]:
        raw_output = await self.client.chat(
            [
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": build_baseline_prompt(example)},
            ],
            temperature=0.0,
        )
        return parse_baseline_classification(raw_output), raw_output


def build_prediction_from_raw_output(
    example: NormalizedExample,
    *,
    raw_output: str,
    latency_ms: float,
    continue_on_error: bool,
) -> BaselinePrediction:
    try:
        result = parse_baseline_classification(raw_output)
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


def parse_baseline_classification(raw_output: str) -> BaselineClassification:
    """Parse and validate the first normalized JSON object produced by the model."""
    payload = _extract_first_json_object(raw_output)
    try:
        return BaselineClassification.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid baseline JSON: {raw_output!r}") from exc


def parse_classification_result(raw_output: str) -> ClassificationResult:
    """Parse and validate the first JSON object produced by the model."""
    payload = _extract_first_json_object(raw_output)
    try:
        return ClassificationResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid classification JSON: {raw_output!r}") from exc


def _extract_first_json_object(raw_output: str) -> dict:
    decoder = json.JSONDecoder()
    text = raw_output.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Model did not return JSON. Output: {raw_output!r}")
