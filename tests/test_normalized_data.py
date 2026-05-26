import pytest
from pydantic import ValidationError

from src.data.jsonl import read_jsonl, validate_jsonl, write_jsonl
from src.domain.schemas import NormalizedExample


def valid_payload() -> dict:
    return {
        "id": "example-001",
        "source_dataset": "synthetic",
        "text": "Necesito ayuda para configurar el bot.",
        "topic": "soporte",
        "risk_labels": ["sin_riesgo"],
        "action": "allow",
        "split": "train",
        "original_labels": {"category": "help"},
        "metadata": {"language": "es"},
    }


def test_create_valid_example() -> None:
    example = NormalizedExample.model_validate(valid_payload())

    assert example.id == "example-001"
    assert example.topic == "soporte"
    assert example.risk_labels == ["sin_riesgo"]


@pytest.mark.parametrize("field", ["id", "source_dataset", "text", "topic"])
def test_rejects_empty_required_text_fields(field: str) -> None:
    payload = valid_payload()
    payload[field] = " "

    with pytest.raises(ValidationError):
        NormalizedExample.model_validate(payload)


def test_rejects_invalid_topic() -> None:
    payload = valid_payload()
    payload["topic"] = "tecnologia"

    with pytest.raises(ValidationError, match="topic"):
        NormalizedExample.model_validate(payload)


def test_rejects_empty_risk_labels() -> None:
    payload = valid_payload()
    payload["risk_labels"] = []

    with pytest.raises(ValidationError, match="risk_labels"):
        NormalizedExample.model_validate(payload)


def test_rejects_invalid_risk_label() -> None:
    payload = valid_payload()
    payload["risk_labels"] = ["acoso"]

    with pytest.raises(ValidationError, match="risk_labels"):
        NormalizedExample.model_validate(payload)


def test_rejects_invalid_action() -> None:
    payload = valid_payload()
    payload["action"] = "warn"

    with pytest.raises(ValidationError, match="action"):
        NormalizedExample.model_validate(payload)


def test_rejects_invalid_split() -> None:
    payload = valid_payload()
    payload["split"] = "dev"

    with pytest.raises(ValidationError, match="split"):
        NormalizedExample.model_validate(payload)


def test_rejects_sin_riesgo_combined_with_other_label() -> None:
    payload = valid_payload()
    payload["risk_labels"] = ["sin_riesgo", "spam_fraude"]

    with pytest.raises(ValidationError, match="sin_riesgo"):
        NormalizedExample.model_validate(payload)


def test_write_and_read_jsonl_preserves_data(tmp_path) -> None:
    path = tmp_path / "normalized.jsonl"
    example = NormalizedExample.model_validate(valid_payload())

    write_jsonl(path, [example])
    records = read_jsonl(path)

    assert [record.model_dump() for record in records] == [example.model_dump()]


def test_write_jsonl_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "processed" / "jigsaw" / "normalized.jsonl"
    example = NormalizedExample.model_validate(valid_payload())

    write_jsonl(path, [example])

    assert read_jsonl(path)[0].model_dump() == example.model_dump()


def test_validate_jsonl_accepts_valid_file(tmp_path) -> None:
    path = tmp_path / "normalized.jsonl"
    payload = valid_payload()

    write_jsonl(path, [payload])

    assert (
        validate_jsonl(path)[0].model_dump()
        == NormalizedExample.model_validate(payload).model_dump()
    )
