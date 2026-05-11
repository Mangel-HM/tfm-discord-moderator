from __future__ import annotations

import asyncio

from tfm_discord_moderator.config import get_settings
from tfm_discord_moderator.inference.llama_cpp_client import LlamaCppClient


async def main() -> None:
    settings = get_settings()
    client = LlamaCppClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    output = await client.chat(
        [
            {"role": "system", "content": "Responde de forma breve."},
            {"role": "user", "content": "Di OK si puedes recibir mensajes."},
        ],
        temperature=0.0,
    )
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
