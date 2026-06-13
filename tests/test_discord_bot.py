from __future__ import annotations

import asyncio
from typing import Any

from src.config import Settings
from src.discord_bot.bot import (
    BufferedChannelMessage,
    ModerationDecision,
    append_buffered_message,
    build_moderation_classifier,
    format_allow_console_line,
    get_context_text,
    format_moderation_notice,
    remove_buffered_message,
    remove_buffered_messages,
)
from src.domain.schemas import DiscordMessage, NormalizedClassification
from src.presentation.labels import display_action, display_risk_label, display_topic


class FakeLlamaClient:
    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        FakeLlamaClient.kwargs = kwargs


class FakeBaselineClassifier:
    def __init__(self, client: FakeLlamaClient, taxonomy: dict):
        self.client = client
        self.taxonomy = taxonomy

    async def classify_normalized_message(
        self, message: DiscordMessage
    ) -> NormalizedClassification:
        return NormalizedClassification(
            topic="otro",
            risk_labels=["insulto_toxicidad"],
            action="review",
            confidence=0.82,
            rationale="Contains an insult.",
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
    assert result.decision.topic == "otro"
    assert result.decision.risk_labels == ["insulto_toxicidad"]
    assert result.decision.confidence == 0.82
    assert result.decision.rationale == "Contains an insult."


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
    assert "- Suggested action: Review" in notice
    assert "- Risks: Insult/toxicity" in notice
    assert "- Topic: Other" in notice
    assert "- Latency: 1540 ms" in notice


def test_formats_optional_baseline_fields_when_present() -> None:
    notice = format_moderation_notice(
        channel_name="general",
        author_name="usuario_demo",
        message_text="You are awful.",
        context_count=1,
        decision=ModerationDecision(
            topic="otro",
            risk_labels=["insulto_toxicidad"],
            action="review",
            confidence=0.82,
            rationale="Contains an insult.",
        ),
        latency_ms=1540.2,
    )

    assert "- Confidence: 0.82" in notice
    assert "- Rationale: Contains an insult." in notice


def test_internal_decision_values_remain_original_ids() -> None:
    decision = ModerationDecision(
        topic="otro",
        risk_labels=["insulto_toxicidad"],
        action="review",
    )

    assert decision.topic == "otro"
    assert decision.risk_labels == ["insulto_toxicidad"]
    assert decision.action == "review"


def test_display_labels_fall_back_to_original_value() -> None:
    assert display_topic("custom_topic") == "custom_topic"
    assert display_risk_label("custom_risk") == "custom_risk"
    assert display_action("custom_action") == "custom_action"


def test_allow_console_line_uses_display_names() -> None:
    line = format_allow_console_line(
        channel_name="general",
        decision=ModerationDecision(
            topic="soporte",
            risk_labels=["sin_riesgo"],
            action="allow",
        ),
        latency_ms=15.2,
    )

    assert "Technical support" in line
    assert "risks=No risk" in line
    assert "soporte" not in line
    assert "sin_riesgo" not in line


def test_deleted_messages_are_removed_from_context_buffer() -> None:
    from collections import deque

    buffer: deque[BufferedChannelMessage] = deque(maxlen=6)
    first_run = [
        (1, "user_a: hi everyone"),
        (2, "user_a: I need help with my setup"),
        (3, "user_a: this channel is for the demo"),
        (4, "user_a: You are awful and nobody wants you here."),
    ]
    for message_id, text in first_run:
        append_buffered_message(buffer, message_id=message_id, text=text)

    remove_buffered_messages(buffer, {message_id for message_id, _ in first_run})

    append_buffered_message(buffer, message_id=5, text="user_a: hi everyone")
    append_buffered_message(buffer, message_id=6, text="user_a: I need help with my setup")
    append_buffered_message(buffer, message_id=7, text="user_a: this channel is for the demo")

    assert get_context_text(buffer) == [
        "user_a: hi everyone",
        "user_a: I need help with my setup",
        "user_a: this channel is for the demo",
    ]


def test_single_delete_removes_only_matching_buffered_message() -> None:
    from collections import deque

    buffer: deque[BufferedChannelMessage] = deque(maxlen=6)
    append_buffered_message(buffer, message_id=1, text="user_a: first")
    append_buffered_message(buffer, message_id=2, text="user_b: second")
    append_buffered_message(buffer, message_id=3, text="user_c: third")

    remove_buffered_message(buffer, message_id=2)

    assert get_context_text(buffer) == ["user_a: first", "user_c: third"]


def test_bulk_delete_ignores_unknown_message_ids() -> None:
    from collections import deque

    buffer: deque[BufferedChannelMessage] = deque(maxlen=6)
    append_buffered_message(buffer, message_id=1, text="user_a: first")
    append_buffered_message(buffer, message_id=2, text="user_b: second")
    append_buffered_message(buffer, message_id=3, text="user_c: third")

    remove_buffered_messages(buffer, {2, 999})

    assert get_context_text(buffer) == ["user_a: first", "user_c: third"]


def test_context_buffer_still_respects_max_length() -> None:
    from collections import deque

    buffer: deque[BufferedChannelMessage] = deque(maxlen=3)
    for message_id in range(5):
        append_buffered_message(
            buffer,
            message_id=message_id,
            text=f"user_a: message {message_id}",
        )

    assert get_context_text(buffer) == [
        "user_a: message 2",
        "user_a: message 3",
        "user_a: message 4",
    ]
