from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.classification.baseline_classifier import BaselineClassifier
from src.config import get_settings
from src.domain.schemas import DiscordMessage
from src.inference.llama_cpp_client import LlamaCppClient


SAMPLE_PATH = Path("data/samples/messages_sample.jsonl")


async def main() -> None:
    settings = get_settings()
    client = LlamaCppClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    classifier = BaselineClassifier(client)

    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        message_fields = {
            key: item[key] for key in ["message_id", "channel", "author_role", "context", "text"]
        }
        message = DiscordMessage(**message_fields)
        result = await classifier.classify_normalized_message(message)
        output = {"message_id": message.message_id, **result.model_dump()}
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
