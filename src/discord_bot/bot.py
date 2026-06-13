from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import discord
from rich.console import Console

from src.classification.baseline_classifier import BaselineClassifier
from src.classification.lora_classifier import LoraModerationClassifier
from src.config import Settings, get_settings
from src.domain.schemas import DiscordMessage, NormalizedClassification
from src.inference.llama_cpp_client import LlamaCppClient
from src.presentation.labels import display_action, display_risk_labels, display_topic

console = Console()


@dataclass(frozen=True)
class ModerationDecision:
    topic: str
    risk_labels: list[str]
    action: str
    confidence: float | None = None
    rationale: str | None = None


@dataclass(frozen=True)
class BufferedChannelMessage:
    message_id: int
    text: str


@dataclass(frozen=True)
class TimedModerationDecision:
    decision: ModerationDecision
    latency_ms: float


class BotModerationClassifier(Protocol):
    async def classify_message(self, message: DiscordMessage) -> TimedModerationDecision:
        pass


class BaselineBotClassifier:
    def __init__(self, classifier: BaselineClassifier):
        self.classifier = classifier

    async def classify_message(self, message: DiscordMessage) -> TimedModerationDecision:
        started = perf_counter()
        result = await self.classifier.classify_normalized_message(message)
        latency_ms = (perf_counter() - started) * 1000
        return TimedModerationDecision(
            decision=decision_from_normalized_result(result),
            latency_ms=latency_ms,
        )


class LoraBotClassifier:
    def __init__(self, classifier: LoraModerationClassifier):
        self.classifier = classifier
        self._lock = asyncio.Lock()

    async def classify_message(self, message: DiscordMessage) -> TimedModerationDecision:
        started = perf_counter()
        async with self._lock:
            result = await asyncio.to_thread(self.classifier.classify_message, message)
        latency_ms = (perf_counter() - started) * 1000
        return TimedModerationDecision(
            decision=decision_from_normalized_result(result),
            latency_ms=latency_ms,
        )


def decision_from_normalized_result(result: NormalizedClassification) -> ModerationDecision:
    return ModerationDecision(
        topic=result.topic,
        risk_labels=result.risk_labels,
        action=result.action,
        confidence=result.confidence,
        rationale=result.rationale,
    )


def build_moderation_classifier(
    settings: Settings,
    *,
    llama_client_factory: Any = LlamaCppClient,
    baseline_classifier_factory: Any = BaselineClassifier,
    lora_classifier_factory: Any = LoraModerationClassifier,
) -> BotModerationClassifier:
    if settings.moderation_backend == "baseline":
        llm_client = llama_client_factory(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        classifier = baseline_classifier_factory(llm_client)
        return BaselineBotClassifier(classifier)

    if not settings.lora_model_name_or_path or not settings.lora_adapter_dir:
        raise RuntimeError("LoRA settings are missing for MODERATION_BACKEND=lora")
    classifier = lora_classifier_factory(
        model_name_or_path=settings.lora_model_name_or_path,
        adapter_dir=settings.lora_adapter_dir,
        max_new_tokens=settings.lora_max_new_tokens,
        temperature=settings.lora_temperature,
        bf16=settings.lora_bf16,
        fp16=settings.lora_fp16,
    )
    return LoraBotClassifier(classifier)


def append_buffered_message(
    buffer: deque[BufferedChannelMessage],
    *,
    message_id: int,
    text: str,
) -> None:
    buffer.append(BufferedChannelMessage(message_id=message_id, text=text))


def get_context_text(buffer: deque[BufferedChannelMessage]) -> list[str]:
    return [entry.text for entry in buffer]


def remove_buffered_message(buffer: deque[BufferedChannelMessage], *, message_id: int) -> None:
    remove_buffered_messages(buffer, {message_id})


def remove_buffered_messages(
    buffer: deque[BufferedChannelMessage],
    message_ids: set[int],
) -> None:
    kept = [entry for entry in buffer if entry.message_id not in message_ids]
    buffer.clear()
    buffer.extend(kept)


def format_moderation_notice(
    *,
    channel_name: str,
    author_name: str,
    message_text: str,
    context_count: int,
    decision: ModerationDecision,
    latency_ms: float,
) -> str:
    confidence = (
        f"\n- Confidence: {decision.confidence:.2f}" if decision.confidence is not None else ""
    )
    rationale = f"\n- Rationale: {decision.rationale}" if decision.rationale else ""
    return f"""
Suggested review

Channel: #{channel_name}
Author: {author_name}
Message:
"{message_text}"

Context used: {context_count} previous messages

Result:
- Suggested action: {display_action(decision.action)}
- Risks: {display_risk_labels(decision.risk_labels)}
- Topic: {display_topic(decision.topic)}
- Latency: {latency_ms:.0f} ms{confidence}{rationale}
""".strip()


def format_allow_console_line(
    *,
    channel_name: str,
    decision: ModerationDecision,
    latency_ms: float,
) -> str:
    return (
        f"[allow] #{channel_name} -> {display_topic(decision.topic)} "
        f"risks={display_risk_labels(decision.risk_labels)} "
        f"latency={latency_ms:.0f}ms"
    )


def build_bot() -> discord.Client:
    settings = get_settings()
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    buffers: dict[int, deque[BufferedChannelMessage]] = defaultdict(
        lambda: deque(maxlen=settings.max_context_messages)
    )

    classifier = build_moderation_classifier(settings)

    @client.event
    async def on_ready() -> None:
        console.print(
            f"Bot connected as {client.user} "
            f"(backend={settings.moderation_backend}, auto_delete={settings.auto_delete})"
        )

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or not message.content.strip():
            return

        channel_id = message.channel.id
        context = get_context_text(buffers[channel_id])
        append_buffered_message(
            buffers[channel_id],
            message_id=message.id,
            text=f"{message.author.display_name}: {message.content}",
        )

        domain_message = DiscordMessage(
            message_id=str(message.id),
            channel=getattr(message.channel, "name", str(channel_id)),
            author_role=message.author.display_name,
            context=context,
            text=message.content,
        )

        try:
            timed_decision = await classifier.classify_message(domain_message)
        except Exception as exc:  # noqa: BLE001 - keep bot alive during experimentation
            console.print(f"[red]Error classifying message {message.id}: {exc}[/red]")
            return

        notice = format_moderation_notice(
            channel_name=domain_message.channel,
            author_name=message.author.display_name,
            message_text=message.content,
            context_count=len(context),
            decision=timed_decision.decision,
            latency_ms=timed_decision.latency_ms,
        )

        if timed_decision.decision.action == "allow":
            console.print(
                format_allow_console_line(
                    channel_name=domain_message.channel,
                    decision=timed_decision.decision,
                    latency_ms=timed_decision.latency_ms,
                )
            )
            return

        console.print(notice)
        if settings.discord_mod_channel_id:
            mod_channel = client.get_channel(settings.discord_mod_channel_id)
            if isinstance(mod_channel, discord.abc.Messageable):
                await mod_channel.send(notice)

    @client.event
    async def on_message_delete(message: discord.Message) -> None:
        remove_buffered_message(buffers[message.channel.id], message_id=message.id)

    @client.event
    async def on_bulk_message_delete(messages: list[discord.Message]) -> None:
        deleted_by_channel: dict[int, set[int]] = defaultdict(set)
        for message in messages:
            deleted_by_channel[message.channel.id].add(message.id)
        for channel_id, deleted_ids in deleted_by_channel.items():
            remove_buffered_messages(buffers[channel_id], deleted_ids)

    return client


def main() -> None:
    settings = get_settings()
    if not settings.discord_token:
        raise RuntimeError("DISCORD_TOKEN is missing. Copy .env.example to .env and fill it.")
    build_bot().run(settings.discord_token)


if __name__ == "__main__":
    main()
