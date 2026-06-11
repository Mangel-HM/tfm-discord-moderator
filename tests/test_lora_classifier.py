from __future__ import annotations

import pytest

from src.classification.lora_classifier import LoraModerationClassifier, build_lora_messages
from src.domain.schemas import ALLOWED_ACTIONS, ALLOWED_RISK_LABELS, ALLOWED_TOPICS, DiscordMessage


def make_message(*, context: list[str] | None = None) -> DiscordMessage:
    return DiscordMessage(
        message_id="discord-1",
        channel="general",
        author_role="usuario_demo",
        context=context or [],
        text="You are awful.",
    )


class FakeLoraClassifier(LoraModerationClassifier):
    def __init__(self, raw_output: str):
        self.raw_output = raw_output

    def generate(self, message: DiscordMessage) -> str:
        return self.raw_output


def test_build_lora_prompt_with_context_lists_allowed_labels() -> None:
    messages = build_lora_messages(make_message(context=["user_a: hello", "user_b: please stop"]))
    prompt = messages[1]["content"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Classify only the current message" in prompt
    assert "Previous channel context:" in prompt
    assert "1. user_a: hello" in prompt
    assert "2. user_b: please stop" in prompt
    assert "Current message to classify:" in prompt
    assert "usuario_demo: You are awful." in prompt
    for topic in ALLOWED_TOPICS:
        assert topic in prompt
    for risk_label in ALLOWED_RISK_LABELS:
        assert risk_label in prompt
    for action in ALLOWED_ACTIONS:
        assert action in prompt
    assert "spam_fraude" not in prompt


def test_build_lora_prompt_without_context_is_explicit() -> None:
    prompt = build_lora_messages(make_message())[1]["content"]

    assert "(no previous channel context)" in prompt
    assert "Current message to classify:" in prompt


def test_lora_classifier_parses_valid_json() -> None:
    classifier = FakeLoraClassifier(
        '{"topic":"otro","risk_labels":["insulto_toxicidad"],"action":"review"}'
    )

    result = classifier.classify_message(make_message())

    assert result.topic == "otro"
    assert result.risk_labels == ["insulto_toxicidad"]
    assert result.action == "review"


def test_lora_classifier_rejects_invalid_json() -> None:
    classifier = FakeLoraClassifier("not json")

    with pytest.raises(ValueError, match="did not return JSON"):
        classifier.classify_message(make_message())
