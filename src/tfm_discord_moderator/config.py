from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = Field(default="http://127.0.0.1:8001/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="not-needed", alias="LLM_API_KEY")
    llm_model: str = Field(default="discord-qwen-local", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=120.0, alias="LLM_TIMEOUT_SECONDS")

    classification_threshold: float = Field(default=0.70, alias="CLASSIFICATION_THRESHOLD")
    max_context_messages: int = Field(default=6, alias="MAX_CONTEXT_MESSAGES")

    discord_token: str | None = Field(default=None, alias="DISCORD_TOKEN")
    discord_mod_channel_id: int | None = Field(default=None, alias="DISCORD_MOD_CHANNEL_ID")
    auto_delete: bool = Field(default=False, alias="AUTO_DELETE")


def get_settings() -> Settings:
    return Settings()
