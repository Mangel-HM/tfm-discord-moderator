from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


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
