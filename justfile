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

# Start the local llama.cpp OpenAI-compatible server using .env settings.
llama-server:
    @if (-not $env:LLAMA_SERVER_PATH) { throw "LLAMA_SERVER_PATH is missing in .env" }; if (-not $env:LLAMA_MODEL_PATH) { throw "LLAMA_MODEL_PATH is missing in .env" }; $alias = if ($env:LLAMA_MODEL_ALIAS) { $env:LLAMA_MODEL_ALIAS } else { "discord-qwen-local" }; $hostName = if ($env:LLAMA_HOST) { $env:LLAMA_HOST } else { "127.0.0.1" }; $port = if ($env:LLAMA_PORT) { $env:LLAMA_PORT } else { "8001" }; $ctxSize = if ($env:LLAMA_CTX_SIZE) { $env:LLAMA_CTX_SIZE } else { "8192" }; $gpuLayers = if ($env:LLAMA_N_GPU_LAYERS) { $env:LLAMA_N_GPU_LAYERS } else { "999" }; & $env:LLAMA_SERVER_PATH --model $env:LLAMA_MODEL_PATH --alias $alias --host $hostName --port $port --ctx-size $ctxSize --n-gpu-layers $gpuLayers

# Classify the synthetic JSONL sample through the local model endpoint.
sample:
    @uv run python scripts/classify_sample.py

# Convert a manually downloaded Jigsaw CSV to normalized JSONL.
prepare-jigsaw INPUT OUTPUT SPLIT="train" *ARGS:
    @uv run python scripts/prepare_jigsaw.py --input "{{INPUT}}" --output "{{OUTPUT}}" --split "{{SPLIT}}" {{ARGS}}

# Run the normalized llama.cpp baseline over a JSONL file.
baseline INPUT OUTPUT MAX_EXAMPLES="" *ARGS:
    @uv run python scripts/run_baseline.py --input "{{INPUT}}" --output "{{OUTPUT}}" {{ if MAX_EXAMPLES != "" { "--max-examples " + MAX_EXAMPLES } else { "" } }} {{ARGS}}

# Run the Discord bot in observation mode.
bot:
    @uv run python -m src.discord_bot.bot
