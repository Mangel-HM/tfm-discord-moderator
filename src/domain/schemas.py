from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ALLOWED_TOPICS = ("gaming", "soporte", "social_general", "otro")
ALLOWED_RISK_LABELS = (
    "sin_riesgo",
    "insulto_toxicidad",
    "odio_discriminacion",
    "amenaza_violencia",
    "sexual_nsfw",
    "spam_fraude",
)
ALLOWED_ACTIONS = ("allow", "review", "warn_candidate", "delete_candidate")
ALLOWED_SPLITS = ("train", "validation", "test")


class ModerationAction(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    WARN = "warn"
    DELETE_CANDIDATE = "delete_candidate"


class DiscordMessage(BaseModel):
    message_id: str
    channel: str
    author_role: str = "unknown"
    context: list[str] = Field(default_factory=list)
    text: str


class ClassificationResult(BaseModel):
    label: str
    action: ModerationAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=500)
    risk: Literal["low", "medium", "high"]

    @field_validator("label")
    @classmethod
    def label_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label cannot be empty")
        return value


class NormalizedExample(BaseModel):
    id: str
    source_dataset: str
    text: str
    topic: str
    risk_labels: list[str]
    action: str
    split: str
    original_labels: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    @field_validator("id", "source_dataset", "text", "topic")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field cannot be empty")
        return value

    @field_validator("topic")
    @classmethod
    def topic_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_TOPICS:
            raise ValueError(f"topic must be one of: {', '.join(ALLOWED_TOPICS)}")
        return value

    @field_validator("risk_labels")
    @classmethod
    def risk_labels_must_be_allowed(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("risk_labels must contain at least one label")
        invalid = [label for label in value if label not in ALLOWED_RISK_LABELS]
        if invalid:
            raise ValueError(f"risk_labels contains invalid labels: {', '.join(invalid)}")
        return value

    @field_validator("action")
    @classmethod
    def action_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(ALLOWED_ACTIONS)}")
        return value

    @field_validator("split")
    @classmethod
    def split_must_be_allowed(cls, value: str) -> str:
        if value not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of: {', '.join(ALLOWED_SPLITS)}")
        return value

    @model_validator(mode="after")
    def sin_riesgo_must_not_be_combined(self) -> "NormalizedExample":
        if "sin_riesgo" in self.risk_labels and len(self.risk_labels) > 1:
            raise ValueError("sin_riesgo cannot be combined with other risk labels")
        return self
