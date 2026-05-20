# TFM Discord Moderator

Prueba de concepto para clasificar mensajes de Discord por tematica y riesgo de moderacion
usando un LLM local servido por llama.cpp.

## Requisitos

```powershell
winget install --id astral-sh.uv -e
winget install --id Casey.Just -e
uv --version
just --version
```

## Preparar entorno

```powershell
just install
just env-init
```

Edita `.env` si necesitas cambiar el endpoint o el modelo local:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=discord-qwen-local
AUTO_DELETE=false
```

No guardes `.env`, tokens, modelos GGUF ni datasets privados en Git.

## llama.cpp

El endpoint esperado es:

```text
http://127.0.0.1:8001/v1/chat/completions
```

Ejemplo en Windows:

```powershell
cd C:\llama.cpp
.\llama-server.exe `
  --model C:\models\qwen3.5-2b\Qwen3.5-2B-Q4_K_M.gguf `
  --alias discord-qwen-local `
  --host 127.0.0.1 `
  --port 8001 `
  --ctx-size 8192 `
  --n-gpu-layers 999
```

## Comandos

```powershell
just check        # lint, typecheck y tests
just env          # informacion local de Python/SO/GPU
just llama-check  # comprueba llama.cpp
just sample       # clasifica ejemplos sinteticos
just bot          # ejecuta el bot en modo observacion
```

`just llama-check` y `just sample` requieren `llama-server` levantado.

## Bot de Discord

Para probar el bot:

1. Crea una aplicacion/bot en Discord.
2. Activa el intent de contenido de mensajes.
3. Configura `DISCORD_TOKEN` en `.env`.
4. Manten `AUTO_DELETE=false`.

```powershell
just bot
```
