from __future__ import annotations

from pathlib import Path

import yaml

from src.domain.schemas import (
    ALLOWED_ACTIONS,
    ALLOWED_RISK_LABELS,
    ALLOWED_TOPICS,
    DiscordMessage,
    NormalizedExample,
)


SYSTEM_PROMPT = """Eres un clasificador de mensajes de Discord para apoyo a moderacion.
Tu tarea es clasificar un mensaje en una tematica y proponer una accion de moderacion conservadora.
No inventes datos. No ejecutes acciones. Devuelve solo JSON valido.
"""

BASELINE_SYSTEM_PROMPT = """You are a conservative Discord moderation classifier.
Classify English messages only. Return only valid JSON and no markdown."""


def load_label_taxonomy(path: str | Path = "configs/labels.yml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_classification_prompt(message: DiscordMessage, taxonomy: dict) -> str:
    labels = "\n".join(
        f"- {item['id']}: {item['description']}" for item in taxonomy.get("labels", [])
    )
    actions = ", ".join(taxonomy.get("moderation_actions", []))
    context = "\n".join(f"- {entry}" for entry in message.context[-6:]) or "(sin contexto previo)"

    return f"""
Clasifica el siguiente mensaje de Discord.

Etiquetas disponibles:
{labels}

Acciones disponibles: {actions}

Contexto reciente del canal:
{context}

Canal: {message.channel}
Rol aproximado del autor: {message.author_role}
Mensaje actual:
{message.text}

Devuelve exclusivamente este JSON, sin markdown:
{{
  "label": "una etiqueta de la lista",
  "action": "allow|review|warn_candidate|delete_candidate",
  "confidence": 0.0,
  "risk": "low|medium|high",
  "rationale": "explicacion breve"
}}
""".strip()


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
- "rationale": a short English explanation

Rules:
- If there is no moderation risk, use risk_labels = ["sin_riesgo"] and action = "allow".
- Do not combine sin_riesgo with any other risk label.
- For Jigsaw examples, the expected topic is usually "otro", but always return a valid topic.
- Do not invent labels outside the allowed lists.
- Prefer "review" for risky content unless the message clearly suggests a warning or deletion candidate.

Message:
{example.text}
""".strip()
