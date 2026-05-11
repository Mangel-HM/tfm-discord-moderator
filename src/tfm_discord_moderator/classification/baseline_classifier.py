from __future__ import annotations

import json
import re

from pydantic import ValidationError

from tfm_discord_moderator.classification.prompts import SYSTEM_PROMPT, build_classification_prompt
from tfm_discord_moderator.domain.schemas import ClassificationResult, DiscordMessage
from tfm_discord_moderator.inference.llama_cpp_client import LlamaCppClient

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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
    match = _JSON_OBJECT_RE.search(raw_output.strip())
    if not match:
        raise ValueError(f"Model did not return JSON. Output: {raw_output!r}")
    try:
        payload = json.loads(match.group(0))
        return ClassificationResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid classification JSON: {raw_output!r}") from exc
