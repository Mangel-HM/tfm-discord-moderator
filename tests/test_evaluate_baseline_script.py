from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_baseline import main


def write_prediction(path: Path, *, pred_topic: str = "gaming") -> None:
    payload = {
        "id": "jigsaw_000001",
        "source_dataset": "jigsaw",
        "text": "A public comment.",
        "gold_topic": "otro",
        "gold_risk_labels": ["sin_riesgo"],
        "gold_action": "allow",
        "pred_topic": pred_topic,
        "pred_risk_labels": ["sin_riesgo"],
        "pred_action": "allow",
        "confidence": 0.9,
        "rationale": "No moderation risk.",
        "latency_ms": 12.5,
        "raw_response": "{}",
        "parse_error": None,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_evaluate_baseline_script_writes_metrics_and_can_ignore_topic(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "metrics.json"
    write_prediction(input_path)

    metrics = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ignore-topic",
        ]
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["action_accuracy"] == 1.0
    assert written["ignore_topic"] is True
    assert "topic_accuracy" not in written
