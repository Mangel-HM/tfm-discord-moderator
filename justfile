set shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true
export UV_CACHE_DIR := ".uv-cache"
export PYTHONPATH := "."
ruff_targets := "src scripts tests pyproject.toml"

default:
    @just --list

# Show available recipes.
help:
    @just --list

# Create .env from the example file.
env-init:
    @if (Test-Path -LiteralPath ".env") { throw ".env already exists" }
    @Copy-Item -LiteralPath ".env.example" -Destination ".env"
    @Write-Host "Created .env from .env.example"

# Upgrade the lockfile and synchronize the development environment.
install:
    @uv lock --upgrade
    @uv sync

# Run the complete local quality suite.
check: lint typecheck test
    @Write-Host "Checks passed."

# Run Ruff formatting and automatic lint fixes.
lint:
    @uv run ruff format {{ruff_targets}}
    @uv run ruff check --fix {{ruff_targets}}

# Run ty type checks.
typecheck:
    @uv run ty check

# Run pytest with optional extra arguments.
test *args:
    @uv run python -m pytest {{args}}

# Run pytest with optional extra arguments and stop on first failure.
test-x *args:
    @uv run python -m pytest -x {{args}}

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
    @uv run python -m src.discord_bot.bot
