from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
