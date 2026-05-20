from __future__ import annotations

from pathlib import Path

import yaml

from src.domain.schemas import DiscordMessage


SYSTEM_PROMPT = """Eres un clasificador de mensajes de Discord para apoyo a moderacion.
Tu tarea es clasificar un mensaje en una tematica y proponer una accion de moderacion conservadora.
No inventes datos. No ejecutes acciones. Devuelve solo JSON valido.
"""


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
  "action": "allow|review|warn|delete_candidate",
  "confidence": 0.0,
  "risk": "low|medium|high",
  "rationale": "explicacion breve"
}}
""".strip()
