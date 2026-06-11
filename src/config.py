from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ModerationBackend = Literal["baseline", "lora"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = Field(default="http://127.0.0.1:8001/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="not-needed", alias="LLM_API_KEY")
    llm_model: str = Field(default="discord-qwen-local", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=120.0, alias="LLM_TIMEOUT_SECONDS")

    moderation_backend: ModerationBackend = Field(default="baseline", alias="MODERATION_BACKEND")
    classification_threshold: float = Field(default=0.70, alias="CLASSIFICATION_THRESHOLD")
    max_context_messages: int = Field(default=6, ge=0, alias="MAX_CONTEXT_MESSAGES")

    lora_model_name_or_path: str | None = Field(default=None, alias="LORA_MODEL_NAME_OR_PATH")
    lora_adapter_dir: str | None = Field(default=None, alias="LORA_ADAPTER_DIR")
    lora_bf16: bool = Field(default=True, alias="LORA_BF16")
    lora_fp16: bool = Field(default=False, alias="LORA_FP16")
    lora_max_new_tokens: int = Field(default=128, ge=1, alias="LORA_MAX_NEW_TOKENS")
    lora_temperature: float = Field(default=0.0, ge=0.0, alias="LORA_TEMPERATURE")

    discord_token: str | None = Field(default=None, alias="DISCORD_TOKEN")
    discord_mod_channel_id: int | None = Field(default=None, alias="DISCORD_MOD_CHANNEL_ID")
    auto_delete: bool = Field(default=False, alias="AUTO_DELETE")

    @model_validator(mode="after")
    def validate_lora_settings(self) -> "Settings":
        if self.lora_bf16 and self.lora_fp16:
            raise ValueError("LORA_BF16 and LORA_FP16 cannot both be true")
        if self.moderation_backend == "lora":
            missing = [
                name
                for name, value in (
                    ("LORA_MODEL_NAME_OR_PATH", self.lora_model_name_or_path),
                    ("LORA_ADAPTER_DIR", self.lora_adapter_dir),
                )
                if value is None or not value.strip()
            ]
            if missing:
                raise ValueError(
                    "Missing required LoRA settings for MODERATION_BACKEND=lora: "
                    + ", ".join(missing)
                )
        return self


def get_settings() -> Settings:
    return Settings()
