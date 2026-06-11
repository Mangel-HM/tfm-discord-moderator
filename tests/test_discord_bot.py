from __future__ import annotations

import asyncio
from typing import Any

from src.config import Settings
from src.discord_bot.bot import (
    ModerationDecision,
    build_moderation_classifier,
    format_moderation_notice,
)
from src.domain.schemas import ClassificationResult, DiscordMessage, NormalizedClassification
from src.domain.schemas import ModerationAction


class FakeLlamaClient:
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        FakeLlamaClient.kwargs = kwargs


class FakeBaselineClassifier:
    def __init__(self, client: FakeLlamaClient, taxonomy: dict):
        self.client = client
        self.taxonomy = taxonomy

    async def classify(self, message: DiscordMessage) -> ClassificationResult:
        return ClassificationResult(
            label="toxicidad_o_conflicto",
            action=ModerationAction.REVIEW,
            confidence=0.82,
            rationale="Contains an insult.",
            risk="medium",
        )


class FakeLoraClassifier:
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        FakeLoraClassifier.kwargs = kwargs

    def classify_message(self, message: DiscordMessage) -> NormalizedClassification:
        return NormalizedClassification(
            topic="otro",
            risk_labels=["insulto_toxicidad"],
            action="review",
        )


def make_message() -> DiscordMessage:
    return DiscordMessage(
        message_id="discord-1",
        channel="general",
        author_role="usuario_demo",
        context=["user_a: hello"],
        text="You are awful.",
    )


def test_builds_baseline_backend_without_lora_settings() -> None:
    settings_factory: Any = Settings
    settings = settings_factory(_env_file=None, MODERATION_BACKEND="baseline")

    classifier = build_moderation_classifier(
        settings,
        llama_client_factory=FakeLlamaClient,
        baseline_classifier_factory=FakeBaselineClassifier,
        taxonomy_loader=lambda: {"labels": [], "moderation_actions": []},
    )
    result = asyncio.run(classifier.classify_message(make_message()))

    assert FakeLlamaClient.kwargs["model"] == "discord-qwen-local"
    assert result.decision.action == "review"
    assert result.decision.topic == "toxicidad_o_conflicto"
    assert result.decision.risk_labels == ["medium"]


def test_builds_lora_backend_with_configured_adapter() -> None:
    settings_factory: Any = Settings
    settings = settings_factory(
        _env_file=None,
        MODERATION_BACKEND="lora",
        LORA_MODEL_NAME_OR_PATH="Qwen/Qwen3.5-2B",
        LORA_ADAPTER_DIR="outputs/lora_jigsaw_balanced_v2",
        LORA_BF16=True,
        LORA_FP16=False,
    )

    classifier = build_moderation_classifier(
        settings,
        lora_classifier_factory=FakeLoraClassifier,
    )
    result = asyncio.run(classifier.classify_message(make_message()))

    assert FakeLoraClassifier.kwargs["model_name_or_path"] == "Qwen/Qwen3.5-2B"
    assert FakeLoraClassifier.kwargs["adapter_dir"] == "outputs/lora_jigsaw_balanced_v2"
    assert result.decision.action == "review"
    assert result.decision.topic == "otro"
    assert result.decision.risk_labels == ["insulto_toxicidad"]


def test_formats_moderation_notice_for_demo() -> None:
    notice = format_moderation_notice(
        channel_name="general",
        author_name="usuario_demo",
        message_text="You are awful.",
        context_count=3,
        decision=ModerationDecision(
            topic="otro",
            risk_labels=["insulto_toxicidad"],
            action="review",
        ),
        latency_ms=1540.2,
    )

    assert "Suggested review" in notice
    assert "Channel: #general" in notice
    assert "Author: usuario_demo" in notice
    assert '"You are awful."' in notice
    assert "Context used: 3 previous messages" in notice
    assert "- Suggested action: review" in notice
    assert "- Risks: insulto_toxicidad" in notice
    assert "- Topic: otro" in notice
    assert "- Latency: 1540 ms" in notice
