# TFM Discord Moderator

Proyecto base para una prueba de concepto de clasificacion tematica y apoyo a moderacion en Discord con un LLM local servido por llama.cpp.

## 1. Preparar entorno Python

En PowerShell, dentro del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
copy .env.example .env
```

## 2. Arrancar llama.cpp server

Ejemplo recomendado para Qwen3.5-2B GGUF en Windows. Ajusta las rutas a tu carpeta real de `llama.cpp` y del modelo:

```powershell
cd C:\llama.cpp
.\llama-server.exe `
  --model C:\models\qwen3.5-2b\Qwen3.5-2B-Q4_K_M.gguf `
  --alias discord-qwen-local `
  --host 127.0.0.1 `
  --port 8001 `
  --ctx-size 8192 `
  --n-gpu-layers 999 `
  --temp 0.7 `
  --top-p 0.8 `
  --top-k 20
```

Si prefieres usar descarga automatica de Hugging Face desde llama.cpp:

```powershell
.\llama-server.exe -hf unsloth/Qwen3.5-2B-GGUF:UD-Q4_K_XL --alias discord-qwen-local --host 127.0.0.1 --port 8001 --ctx-size 8192 --n-gpu-layers 999
```

## 3. Comprobar que el servidor responde

En otra terminal, con el entorno Python activado:

```powershell
python scripts/check_llama_server.py
python scripts/classify_sample.py
pytest
```

## 4. Ejecutar bot de Discord en modo observacion

1. Crea una aplicacion/bot en el portal de Discord.
2. Activa el intent de contenido de mensajes para tu bot de pruebas.
3. Copia el token en `.env` como `DISCORD_TOKEN`.
4. Deja `AUTO_DELETE=false`.

```powershell
python -m tfm_discord_moderator.discord_bot.bot
```

## 5. Siguientes pasos

- Sustituir `data/samples/messages_sample.jsonl` por un conjunto de evaluacion anonimizado.
- Medir latencia, precision, falsos positivos y falsos negativos.
- Ajustar taxonomia de etiquetas.
- Preparar el dataset de entrenamiento en una fase posterior.
