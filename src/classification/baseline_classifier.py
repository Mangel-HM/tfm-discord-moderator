from __future__ import annotations

import json

from pydantic import ValidationError

from src.classification.prompts import SYSTEM_PROMPT, build_classification_prompt
from src.domain.schemas import ClassificationResult, DiscordMessage
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
