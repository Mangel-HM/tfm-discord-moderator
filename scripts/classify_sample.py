from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tfm_discord_moderator.classification.baseline_classifier import BaselineClassifier
from tfm_discord_moderator.classification.prompts import load_label_taxonomy
from tfm_discord_moderator.config import get_settings
from tfm_discord_moderator.domain.schemas import DiscordMessage
from tfm_discord_moderator.inference.llama_cpp_client import LlamaCppClient


async def main() -> None:
    settings = get_settings()
    client = LlamaCppClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    classifier = BaselineClassifier(client, load_label_taxonomy())

    sample_path = Path("data/samples/messages_sample.jsonl")
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        message = DiscordMessage(**{k: item[k] for k in ["message_id", "channel", "author_role", "context", "text"]})
        result = await classifier.classify(message)
        print(json.dumps({"message_id": message.message_id, **result.model_dump()}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
