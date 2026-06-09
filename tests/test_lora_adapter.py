from __future__ import annotations

import pytest

from scripts.test_lora_adapter import build_parser, build_test_messages
from src.domain.schemas import ALLOWED_ACTIONS, ALLOWED_RISK_LABELS, ALLOWED_TOPICS


def test_adapter_test_prompt_lists_allowed_labels() -> None:
    messages = build_test_messages("You are awful.")
    prompt = messages[1]["content"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    for topic in ALLOWED_TOPICS:
        assert topic in prompt
    for risk_label in ALLOWED_RISK_LABELS:
        assert risk_label in prompt
    for action in ALLOWED_ACTIONS:
        assert action in prompt
    assert "You are awful." in prompt


def test_adapter_parser_rejects_incompatible_precision_flags() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--adapter-dir",
                "outputs/lora",
                "--model-name-or-path",
                "Qwen/Qwen3-0.6B",
                "--bf16",
                "--fp16",
            ]
        )
