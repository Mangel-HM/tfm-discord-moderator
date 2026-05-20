import pytest

from src.evaluation.metrics import evaluate_labels


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
