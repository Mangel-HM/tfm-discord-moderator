from __future__ import annotations


RISK_LABEL_DISPLAY_NAMES = {
    "sin_riesgo": "No risk",
    "insulto_toxicidad": "Insult/toxicity",
    "odio_discriminacion": "Hate/discrimination",
    "amenaza_violencia": "Threat/violence",
    "sexual_nsfw": "Sexual/NSFW",
}

TOPIC_DISPLAY_NAMES = {
    "gaming": "Gaming",
    "soporte": "Technical support",
    "social_general": "General social",
    "otro": "Other",
}

ACTION_DISPLAY_NAMES = {
    "allow": "Allow",
    "review": "Review",
    "warn_candidate": "Warning candidate",
    "delete_candidate": "Deletion candidate",
}


def display_risk_label(value: str) -> str:
    return RISK_LABEL_DISPLAY_NAMES.get(value, value)


def display_topic(value: str) -> str:
    return TOPIC_DISPLAY_NAMES.get(value, value)


def display_action(value: str) -> str:
    return ACTION_DISPLAY_NAMES.get(value, value)


def display_risk_labels(values: list[str]) -> str:
    if not values:
        return display_risk_label("sin_riesgo")
    return ", ".join(display_risk_label(value) for value in values)
