import pytest

from src.classification.baseline_classifier import parse_baseline_classification
from src.domain.schemas import BaselinePrediction, NormalizedExample


def valid_example() -> NormalizedExample:
    return NormalizedExample.model_validate(
        {
            "id": "jigsaw_000001",
            "source_dataset": "jigsaw",
            "text": "A harmless public comment.",
            "topic": "otro",
            "risk_labels": ["sin_riesgo"],
            "action": "allow",
            "split": "train",
            "original_labels": {"toxic": 0},
            "metadata": {},
        }
    )


def test_parse_clean_json() -> None:
    raw = (
        '{"topic":"otro","risk_labels":["sin_riesgo"],"action":"allow",'
        '"confidence":0.91,"rationale":"No moderation risk."}'
    )
    result = parse_baseline_classification(raw)
    assert result.topic == "otro"
    assert result.risk_labels == ["sin_riesgo"]
    assert result.action == "allow"
    assert result.confidence == 0.91


def test_parse_json_inside_text() -> None:
    raw = (
        'Result:\n{"topic":"otro","risk_labels":["insulto_toxicidad"],'
        '"action":"review","confidence":0.8,"rationale":"Contains an insult."}'
    )
    result = parse_baseline_classification(raw)
    assert result.topic == "otro"
    assert result.risk_labels == ["insulto_toxicidad"]
    assert result.action == "review"


def test_parse_skips_invalid_brace_before_valid_json() -> None:
    raw = (
        'Note {not json}\n{"topic":"otro","risk_labels":["spam_fraude"],'
        '"action":"review","confidence":0.7,"rationale":"Promotional scam."}'
    )
    result = parse_baseline_classification(raw)
    assert result.risk_labels == ["spam_fraude"]


def test_parse_rejects_missing_json() -> None:
    with pytest.raises(ValueError, match="did not return JSON"):
        parse_baseline_classification("I cannot classify this message.")


def test_parse_rejects_invalid_label() -> None:
    raw = (
        '{"topic":"otro","risk_labels":["harassment"],"action":"review",'
        '"confidence":0.7,"rationale":"Invalid label."}'
    )

    with pytest.raises(ValueError, match="Invalid baseline JSON"):
        parse_baseline_classification(raw)


def test_parse_rejects_sin_riesgo_combined_with_other_label() -> None:
    raw = (
        '{"topic":"otro","risk_labels":["sin_riesgo","insulto_toxicidad"],'
        '"action":"review","confidence":0.7,"rationale":"Contradictory labels."}'
    )

    with pytest.raises(ValueError, match="Invalid baseline JSON"):
        parse_baseline_classification(raw)


def test_create_valid_baseline_prediction() -> None:
    prediction = BaselinePrediction.from_example(
        valid_example(),
        pred_topic="otro",
        pred_risk_labels=["sin_riesgo"],
        pred_action="allow",
        confidence=0.91,
        rationale="No moderation risk.",
        latency_ms=12.5,
        raw_response='{"topic":"otro"}',
    )

    assert prediction.id == "jigsaw_000001"
    assert prediction.gold_risk_labels == ["sin_riesgo"]
    assert prediction.pred_action == "allow"
    assert prediction.parse_error is None


def test_create_parse_error_prediction() -> None:
    prediction = BaselinePrediction.from_parse_error(
        valid_example(),
        latency_ms=10.0,
        raw_response="not json",
        parse_error="Model did not return JSON",
    )

    assert prediction.pred_topic is None
    assert prediction.pred_risk_labels == []
    assert prediction.pred_action is None
    assert prediction.confidence is None
    assert prediction.parse_error == "Model did not return JSON"
