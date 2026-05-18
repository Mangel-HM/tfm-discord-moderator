from __future__ import annotations

from collections import defaultdict, deque

import discord
from rich.console import Console

from tfm_discord_moderator.classification.baseline_classifier import BaselineClassifier
from tfm_discord_moderator.classification.prompts import load_label_taxonomy
from tfm_discord_moderator.config import get_settings
from tfm_discord_moderator.domain.schemas import DiscordMessage, ModerationAction
from tfm_discord_moderator.inference.llama_cpp_client import LlamaCppClient

console = Console()


def build_bot() -> discord.Client:
    settings = get_settings()
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    buffers: dict[int, deque[str]] = defaultdict(
        lambda: deque(maxlen=settings.max_context_messages)
    )

    llm_client = LlamaCppClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    classifier = BaselineClassifier(llm_client, load_label_taxonomy())

    @client.event
    async def on_ready() -> None:
        console.print(f"Bot conectado como {client.user}")

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or not message.content.strip():
            return

        channel_id = message.channel.id
        context = list(buffers[channel_id])
        buffers[channel_id].append(f"{message.author.display_name}: {message.content}")

        domain_message = DiscordMessage(
            message_id=str(message.id),
            channel=getattr(message.channel, "name", str(channel_id)),
            author_role="member",
            context=context,
            text=message.content,
        )

        try:
            result = await classifier.classify(domain_message)
        except Exception as exc:  # noqa: BLE001 - keep bot alive during experimentation
            console.print(f"[red]Error clasificando mensaje {message.id}: {exc}[/red]")
            return

        console.print(
            f"[{result.risk}] #{domain_message.channel} -> {result.label} "
            f"{result.action} conf={result.confidence:.2f} :: {result.rationale}"
        )

        if settings.discord_mod_channel_id and result.action != ModerationAction.ALLOW:
            mod_channel = client.get_channel(settings.discord_mod_channel_id)
            if isinstance(mod_channel, discord.abc.Messageable):
                await mod_channel.send(
                    f"Revision sugerida para mensaje {message.id}: "
                    f"label={result.label}, action={result.action}, "
                    f"confidence={result.confidence:.2f}, risk={result.risk}.\n"
                    f"Razon: {result.rationale}"
                )

    return client


def main() -> None:
    settings = get_settings()
    if not settings.discord_token:
        raise RuntimeError("DISCORD_TOKEN is missing. Copy .env.example to .env and fill it.")
    build_bot().run(settings.discord_token)


if __name__ == "__main__":
    main()
