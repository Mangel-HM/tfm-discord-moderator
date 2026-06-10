set shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true
export UV_CACHE_DIR := ".uv-cache"
export PYTHONPATH := "."
export PYTHONUTF8 := "1"
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
    @$env:UV_TORCH_BACKEND = "cu128"; uv lock --upgrade
    @$env:UV_TORCH_BACKEND = "cu128"; uv sync

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

# Sample a normalized JSONL file by risk label.
sample-normalized INPUT OUTPUT MAX_PER_LABEL *ARGS:
    @uv run python scripts/sample_normalized_dataset.py --input "{{INPUT}}" --output "{{OUTPUT}}" --max-per-label "{{MAX_PER_LABEL}}" {{ARGS}}

# Build a chat/SFT JSONL file from normalized examples.
build-sft INPUT OUTPUT *ARGS:
    @uv run python scripts/build_sft_dataset.py --input "{{INPUT}}" --output "{{OUTPUT}}" {{ARGS}}

# Train a LoRA smoke-test adapter from a chat/SFT JSONL file.
train-lora TRAIN_FILE OUTPUT_DIR *ARGS:
    @uv run python scripts/train_lora.py --train-file "{{TRAIN_FILE}}" --output-dir "{{OUTPUT_DIR}}" {{ARGS}}

# Load a LoRA adapter and generate one test response.
[arg("model_name_or_path", long="model-name-or-path")]
[arg("message", long)]
[arg("max_new_tokens", long="max-new-tokens")]
[arg("temperature", long)]
[arg("bf16", long, value="--bf16")]
[arg("fp16", long, value="--fp16")]
test-lora-adapter ADAPTER_DIR model_name_or_path message="" max_new_tokens="" temperature="" bf16="" fp16="":
    @uv run python scripts/test_lora_adapter.py --adapter-dir "{{ADAPTER_DIR}}" --model-name-or-path "{{model_name_or_path}}" {{bf16}} {{fp16}} {{if message == "" { "" } else { "--message '" + replace(message, "'", "''") + "'" }}} {{if max_new_tokens == "" { "" } else { "--max-new-tokens " + max_new_tokens }}} {{if temperature == "" { "" } else { "--temperature " + temperature }}}

# Run LoRA adapter inference over normalized examples.
lora-adapter INPUT OUTPUT ADAPTER_DIR *ARGS:
    @uv run python scripts/run_lora_adapter.py --input "{{INPUT}}" --output "{{OUTPUT}}" --adapter-dir "{{ADAPTER_DIR}}" {{ARGS}}

# Run the normalized llama.cpp baseline over a JSONL file.
baseline INPUT OUTPUT *ARGS:
    @uv run python scripts/run_baseline.py --input "{{INPUT}}" --output "{{OUTPUT}}" {{ARGS}}

# Evaluate a baseline predictions JSONL file.
evaluate-baseline INPUT OUTPUT *ARGS:
    @uv run python scripts/evaluate_baseline.py --input "{{INPUT}}" --output "{{OUTPUT}}" {{ARGS}}

# Run the Discord bot in observation mode.
bot:
    @uv run python -m src.discord_bot.bot
