import pytest

from src.domain.schemas import BaselinePrediction
from src.evaluation.metrics import evaluate_baseline_predictions, evaluate_labels


def test_evaluate_labels_returns_accuracy_and_confusion() -> None:
    summary = evaluate_labels(
        expected=["soporte_tecnico", "spam_o_promocion", "soporte_tecnico"],
        predicted=["soporte_tecnico", "soporte_tecnico", "soporte_tecnico"],
    )

    assert summary.total == 3
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.confusion[("spam_o_promocion", "soporte_tecnico")] == 1


def test_evaluate_labels_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate_labels(expected=["a"], predicted=[])


def prediction(
    *,
    example_id: str,
    gold_risk_labels: list[str],
    pred_risk_labels: list[str],
    gold_action: str = "review",
    pred_action: str | None = "review",
    gold_topic: str = "otro",
    pred_topic: str | None = "otro",
    latency_ms: float = 10.0,
    parse_error: str | None = None,
) -> BaselinePrediction:
    return BaselinePrediction.model_validate(
        {
            "id": example_id,
            "source_dataset": "jigsaw",
            "text": "A public comment.",
            "gold_topic": gold_topic,
            "gold_risk_labels": gold_risk_labels,
            "gold_action": gold_action,
            "pred_topic": pred_topic,
            "pred_risk_labels": pred_risk_labels,
            "pred_action": pred_action,
            "confidence": None if parse_error else 0.8,
            "rationale": None if parse_error else "Short rationale.",
            "latency_ms": latency_ms,
            "raw_response": "{}",
            "parse_error": parse_error,
        }
    )


def test_evaluate_baseline_predictions_calculates_action_accuracy() -> None:
    metrics = evaluate_baseline_predictions(
        [
            prediction(
                example_id="one",
                gold_risk_labels=["sin_riesgo"],
                pred_risk_labels=["sin_riesgo"],
                gold_action="allow",
                pred_action="allow",
            ),
            prediction(
                example_id="two",
                gold_risk_labels=["insulto_toxicidad"],
                pred_risk_labels=["insulto_toxicidad"],
                gold_action="review",
                pred_action="allow",
            ),
        ]
    )

    assert metrics["total_examples"] == 2
    assert metrics["parsed_predictions"] == 2
    assert metrics["action_accuracy"] == pytest.approx(0.5)


def test_evaluate_baseline_predictions_calculates_risk_exact_match_as_sets() -> None:
    metrics = evaluate_baseline_predictions(
        [
            prediction(
                example_id="one",
                gold_risk_labels=["insulto_toxicidad", "amenaza_violencia"],
                pred_risk_labels=["amenaza_violencia", "insulto_toxicidad"],
            ),
            prediction(
                example_id="two",
                gold_risk_labels=["odio_discriminacion"],
                pred_risk_labels=["insulto_toxicidad"],
            ),
        ]
    )

    assert metrics["risk_exact_match_accuracy"] == pytest.approx(0.5)


def test_evaluate_baseline_predictions_calculates_per_label_precision_recall_f1() -> None:
    metrics = evaluate_baseline_predictions(
        [
            prediction(
                example_id="one",
                gold_risk_labels=["insulto_toxicidad"],
                pred_risk_labels=["insulto_toxicidad"],
            ),
            prediction(
                example_id="two",
                gold_risk_labels=["insulto_toxicidad"],
                pred_risk_labels=["amenaza_violencia"],
            ),
            prediction(
                example_id="three",
                gold_risk_labels=["sin_riesgo"],
                pred_risk_labels=["insulto_toxicidad"],
                gold_action="allow",
            ),
        ]
    )

    insult = metrics["risk_labels"]["insulto_toxicidad"]
    threat = metrics["risk_labels"]["amenaza_violencia"]
    assert insult["true_positives"] == 1
    assert insult["false_positives"] == 1
    assert insult["false_negatives"] == 1
    assert insult["precision"] == pytest.approx(0.5)
    assert insult["recall"] == pytest.approx(0.5)
    assert insult["f1"] == pytest.approx(0.5)
    assert threat["false_positives"] == 1
    assert metrics["macro_f1"] == pytest.approx(0.25)


def test_evaluate_baseline_predictions_handles_parse_errors_separately() -> None:
    metrics = evaluate_baseline_predictions(
        [
            prediction(
                example_id="parsed",
                gold_risk_labels=["sin_riesgo"],
                pred_risk_labels=["sin_riesgo"],
                gold_action="allow",
                pred_action="allow",
            ),
            prediction(
                example_id="bad-json",
                gold_risk_labels=["insulto_toxicidad"],
                pred_risk_labels=[],
                gold_action="review",
                pred_action=None,
                pred_topic=None,
                parse_error="Model did not return JSON",
            ),
        ]
    )

    assert metrics["parse_errors"] == 1
    assert metrics["parsed_predictions"] == 1
    assert metrics["valid_json_rate"] == pytest.approx(0.5)
    assert metrics["action_accuracy"] == pytest.approx(1.0)


def test_evaluate_baseline_predictions_calculates_latency_summary() -> None:
    metrics = evaluate_baseline_predictions(
        [
            prediction(
                example_id="one",
                gold_risk_labels=["sin_riesgo"],
                pred_risk_labels=["sin_riesgo"],
                latency_ms=10.0,
            ),
            prediction(
                example_id="two",
                gold_risk_labels=["sin_riesgo"],
                pred_risk_labels=[],
                latency_ms=40.0,
                pred_action=None,
                pred_topic=None,
                parse_error="Invalid baseline JSON",
            ),
        ]
    )

    assert metrics["average_latency_ms"] == pytest.approx(25.0)
    assert metrics["min_latency_ms"] == pytest.approx(10.0)
    assert metrics["max_latency_ms"] == pytest.approx(40.0)


def test_evaluate_baseline_predictions_can_ignore_topic() -> None:
    metrics = evaluate_baseline_predictions(
        [
            prediction(
                example_id="one",
                gold_risk_labels=["sin_riesgo"],
                pred_risk_labels=["sin_riesgo"],
                gold_topic="otro",
                pred_topic="gaming",
            )
        ],
        ignore_topic=True,
    )

    assert metrics["ignore_topic"] is True
    assert "topic_accuracy" not in metrics
