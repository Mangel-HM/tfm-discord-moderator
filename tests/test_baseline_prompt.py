from src.classification.prompts import build_baseline_prompt
from src.domain.schemas import NormalizedExample


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
