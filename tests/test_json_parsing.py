from tfm_discord_moderator.classification.baseline_classifier import parse_classification_result
from tfm_discord_moderator.domain.schemas import ModerationAction


def test_parse_clean_json() -> None:
    raw = '{"label":"soporte_tecnico","action":"allow","confidence":0.91,"risk":"low","rationale":"Pregunta tecnica."}'
    result = parse_classification_result(raw)
    assert result.label == "soporte_tecnico"
    assert result.action == ModerationAction.ALLOW
    assert result.confidence == 0.91


def test_parse_json_inside_text() -> None:
    raw = 'Resultado:\n{"label":"spam_o_promocion","action":"review","confidence":0.8,"risk":"medium","rationale":"Promocion repetitiva."}'
    result = parse_classification_result(raw)
    assert result.label == "spam_o_promocion"
    assert result.action == ModerationAction.REVIEW
