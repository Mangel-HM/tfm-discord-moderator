from pathlib import Path

import pytest

from tfm_discord_moderator.config import Settings


@pytest.fixture
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "CLASSIFICATION_THRESHOLD",
        "MAX_CONTEXT_MESSAGES",
        "DISCORD_TOKEN",
        "DISCORD_MOD_CHANNEL_ID",
        "AUTO_DELETE",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_env_example_loads_with_safe_defaults(clean_settings_env: None) -> None:
    settings = Settings(_env_file=Path(".env.example"))  # type: ignore[call-arg]

    assert settings.llm_base_url == "http://127.0.0.1:8001/v1"
    assert settings.llm_model == "discord-qwen-local"
    assert settings.auto_delete is False
    assert settings.discord_token is None
    assert settings.discord_mod_channel_id is None
