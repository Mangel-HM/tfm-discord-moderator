from __future__ import annotations

from src.domain.schemas import (
    ALLOWED_ACTIONS,
    ALLOWED_RISK_LABELS,
    ALLOWED_TOPICS,
    DiscordMessage,
    NormalizedExample,
)


BASELINE_SYSTEM_PROMPT = """You are a conservative Discord moderation classifier.
Classify English messages only. Return only valid JSON and no markdown."""


def build_baseline_prompt(example: NormalizedExample) -> str:
    topics = ", ".join(ALLOWED_TOPICS)
    risk_labels = ", ".join(ALLOWED_RISK_LABELS)
    actions = ", ".join(ALLOWED_ACTIONS)

    return f"""
Classify this English message for a Discord moderation baseline.

Return only valid JSON with exactly these fields:
- "topic": one of [{topics}]
- "risk_labels": a non-empty list using only [{risk_labels}]
- "action": one of [{actions}]
- "confidence": a number between 0 and 1
- "rationale": a short English explanation under 300 characters

Rules:
- If there is no moderation risk, use risk_labels = ["sin_riesgo"] and action = "allow".
- Do not combine sin_riesgo with any other risk label.
- For Jigsaw examples, the expected topic is usually "otro", but always return a valid topic.
- Do not invent labels outside the allowed lists.
- Prefer "review" for risky content unless the message clearly suggests a warning or deletion candidate.

Message:
{example.text}
""".strip()


def build_baseline_message_prompt(message: DiscordMessage) -> str:
    topics = ", ".join(ALLOWED_TOPICS)
    risk_labels = ", ".join(ALLOWED_RISK_LABELS)
    actions = ", ".join(ALLOWED_ACTIONS)
    context = "\n".join(f"{index}. {entry}" for index, entry in enumerate(message.context, start=1))
    if not context:
        context = "(no previous channel context)"

    return f"""
Classify the current English Discord message for a moderation baseline.

Return only valid JSON with exactly these fields:
- "topic": one of [{topics}]
- "risk_labels": a non-empty list using only [{risk_labels}]
- "action": one of [{actions}]
- "confidence": a number between 0 and 1
- "rationale": a short English explanation under 300 characters

Rules:
- Classify only the current message.
- Use previous channel context only as supporting context.
- If there is no moderation risk, use risk_labels = ["sin_riesgo"] and action = "allow".
- Do not combine sin_riesgo with any other risk label.
- Do not invent labels outside the allowed lists.
- Prefer "review" for risky content unless the message clearly suggests a warning or deletion candidate.

Previous channel context:
{context}

Current message to classify:
{message.author_role}: {message.text}
""".strip()
