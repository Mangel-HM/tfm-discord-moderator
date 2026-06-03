from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Sequence

from src.domain.schemas import ALLOWED_RISK_LABELS, BaselinePrediction


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    accuracy: float
    confusion: dict[tuple[str, str], int]


def evaluate_labels(expected: list[str], predicted: list[str]) -> EvaluationSummary:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    total = len(expected)
    correct = sum(1 for exp, pred in zip(expected, predicted, strict=True) if exp == pred)
    confusion = Counter(zip(expected, predicted, strict=True))
    return EvaluationSummary(
        total=total,
        accuracy=(correct / total if total else 0.0),
        confusion=dict(confusion),
    )


def evaluate_baseline_predictions(
    predictions: Sequence[BaselinePrediction],
    *,
    ignore_topic: bool = False,
) -> dict:
    total_examples = len(predictions)
    parsed = [prediction for prediction in predictions if prediction.parse_error is None]
    parsed_predictions = len(parsed)
    parse_errors = total_examples - parsed_predictions

    metrics = {
        "total_examples": total_examples,
        "parsed_predictions": parsed_predictions,
        "parse_errors": parse_errors,
        "valid_json_rate": _safe_divide(parsed_predictions, total_examples),
        "ignore_topic": ignore_topic,
        "action_accuracy": _accuracy(
            [prediction.gold_action == prediction.pred_action for prediction in parsed]
        ),
        "risk_exact_match_accuracy": _accuracy(
            [
                set(prediction.gold_risk_labels) == set(prediction.pred_risk_labels)
                for prediction in parsed
            ]
        ),
        "risk_labels": _risk_label_metrics(parsed),
        "average_latency_ms": _average([prediction.latency_ms for prediction in predictions]),
        "min_latency_ms": min((prediction.latency_ms for prediction in predictions), default=0.0),
        "max_latency_ms": max((prediction.latency_ms for prediction in predictions), default=0.0),
    }
    if not ignore_topic:
        metrics["topic_accuracy"] = _accuracy(
            [prediction.gold_topic == prediction.pred_topic for prediction in parsed]
        )
    metrics["macro_f1"] = _macro_f1(metrics["risk_labels"])
    return metrics


def _risk_label_metrics(
    predictions: Sequence[BaselinePrediction],
) -> dict[str, dict[str, float | int]]:
    label_metrics: dict[str, dict[str, float | int]] = {}
    for label in ALLOWED_RISK_LABELS:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for prediction in predictions:
            gold_labels = set(prediction.gold_risk_labels)
            pred_labels = set(prediction.pred_risk_labels)
            if label in gold_labels and label in pred_labels:
                true_positives += 1
            elif label not in gold_labels and label in pred_labels:
                false_positives += 1
            elif label in gold_labels and label not in pred_labels:
                false_negatives += 1

        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        label_metrics[label] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "gold_support": true_positives + false_negatives,
            "predicted_support": true_positives + false_positives,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    return label_metrics


def _macro_f1(label_metrics: dict[str, dict[str, float | int]]) -> float:
    supported_f1 = [values["f1"] for values in label_metrics.values() if values["gold_support"] > 0]
    return _average(supported_f1)


def _accuracy(results: Sequence[bool]) -> float:
    return _safe_divide(sum(results), len(results))


def _average(values: Sequence[float]) -> float:
    return _safe_divide(sum(values), len(values))


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2 * precision * recall, precision + recall)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
