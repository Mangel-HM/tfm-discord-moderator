from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.config import Settings


def load_settings_without_env_file() -> Settings:
    settings_factory: Any = Settings
    return settings_factory(_env_file=None)


@pytest.fixture
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "MODERATION_BACKEND",
        "CLASSIFICATION_THRESHOLD",
        "MAX_CONTEXT_MESSAGES",
        "LORA_MODEL_NAME_OR_PATH",
        "LORA_ADAPTER_DIR",
        "LORA_BF16",
        "LORA_FP16",
        "LORA_MAX_NEW_TOKENS",
        "LORA_TEMPERATURE",
        "DISCORD_TOKEN",
        "DISCORD_MOD_CHANNEL_ID",
        "AUTO_DELETE",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_env_example_loads_with_safe_defaults(clean_settings_env: None) -> None:
    settings_factory: Any = Settings
    settings = settings_factory(_env_file=Path(".env.example"))

    assert settings.llm_base_url == "http://127.0.0.1:8001/v1"
    assert settings.llm_model == "discord-qwen-local"
    assert settings.moderation_backend == "baseline"
    assert settings.auto_delete is False
    assert settings.discord_token is None
    assert settings.discord_mod_channel_id is None


def test_accepts_lora_backend_when_required_settings_are_present(
    clean_settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODERATION_BACKEND", "lora")
    monkeypatch.setenv("LORA_MODEL_NAME_OR_PATH", "Qwen/Qwen3.5-2B")
    monkeypatch.setenv("LORA_ADAPTER_DIR", "outputs/lora_jigsaw_balanced_v2")

    settings = load_settings_without_env_file()

    assert settings.moderation_backend == "lora"
    assert settings.lora_model_name_or_path == "Qwen/Qwen3.5-2B"
    assert settings.lora_adapter_dir == "outputs/lora_jigsaw_balanced_v2"


def test_rejects_invalid_moderation_backend(
    clean_settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODERATION_BACKEND", "remote")

    with pytest.raises(ValidationError, match="MODERATION_BACKEND"):
        load_settings_without_env_file()


def test_lora_backend_requires_model_and_adapter_settings(
    clean_settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODERATION_BACKEND", "lora")

    with pytest.raises(ValidationError, match="LORA_MODEL_NAME_OR_PATH"):
        load_settings_without_env_file()


def test_rejects_lora_bf16_and_fp16_together(
    clean_settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LORA_BF16", "true")
    monkeypatch.setenv("LORA_FP16", "true")

    with pytest.raises(ValidationError, match="LORA_BF16 and LORA_FP16"):
        load_settings_without_env_file()
