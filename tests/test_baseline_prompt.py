from src.classification.prompts import build_baseline_message_prompt, build_baseline_prompt
from src.domain.schemas import DiscordMessage, NormalizedExample


def test_build_baseline_prompt_lists_schema_labels_and_message() -> None:
    example = NormalizedExample.model_validate(
        {
            "id": "jigsaw_000001",
            "source_dataset": "jigsaw",
            "text": "You are awful.",
            "topic": "otro",
            "risk_labels": ["insulto_toxicidad"],
            "action": "review",
            "split": "train",
            "original_labels": {"toxic": 1},
            "metadata": {},
        }
    )

    prompt = build_baseline_prompt(example)

    assert "English" in prompt
    assert "valid JSON" in prompt
    assert '"topic"' in prompt
    assert "gaming" in prompt
    assert "insulto_toxicidad" in prompt
    assert "delete_candidate" in prompt
    assert "Do not combine sin_riesgo" in prompt
    assert "You are awful." in prompt


def test_build_baseline_message_prompt_uses_normalized_bot_contract() -> None:
    message = DiscordMessage(
        message_id="discord-1",
        channel="general",
        author_role="usuario_demo",
        context=["user_a: hello", "user_b: please stop"],
        text="You are awful.",
    )

    prompt = build_baseline_message_prompt(message)

    assert '"topic"' in prompt
    assert '"risk_labels"' in prompt
    assert '"action"' in prompt
    assert '"confidence"' in prompt
    assert '"rationale"' in prompt
    assert "gaming" in prompt
    assert "insulto_toxicidad" in prompt
    assert "delete_candidate" in prompt
    assert "1. user_a: hello" in prompt
    assert "2. user_b: please stop" in prompt
    assert "usuario_demo: You are awful." in prompt
