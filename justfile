set shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true
export UV_CACHE_DIR := ".uv-cache"

default:
    @just --list

# Prepare the local development environment without starting long-running services.
dev:
    @uv sync
    @if (Test-Path -LiteralPath ".env") { Write-Host ".env already exists; leaving it unchanged." }
    @if (-not (Test-Path -LiteralPath ".env")) { Copy-Item -LiteralPath ".env.example" -Destination ".env"; Write-Host "Created .env from .env.example." }
    @Write-Host ""
    @Write-Host "Next commands:"
    @Write-Host "  just check        # run tests, lint and type checks"
    @Write-Host "  just env          # print Python, OS and GPU information"
    @Write-Host "  just llama-check  # check local llama.cpp server"
    @Write-Host "  just sample       # classify sample JSONL messages"
    @Write-Host "  just bot          # run Discord bot in observation mode"

# Upgrade the lockfile and synchronize the editable development environment.
install:
    @uv lock --upgrade
    @uv sync

# Run the complete local quality suite.
check: lint typecheck test

# Run Ruff linting.
lint:
    @uv run ruff check .

# Run mypy over source, scripts and tests.
typecheck:
    @uv run mypy src scripts tests

# Run pytest.
test:
    @uv run pytest

# Print local environment and GPU information.
env:
    @uv run python scripts/check_environment.py

# Check the local llama.cpp OpenAI-compatible server.
llama-check:
    @uv run python scripts/check_llama_server.py

# Classify the synthetic JSONL sample through the local model endpoint.
sample:
    @uv run python scripts/classify_sample.py

# Run the Discord bot in observation mode.
bot:
    @uv run python -m tfm_discord_moderator.discord_bot.bot
