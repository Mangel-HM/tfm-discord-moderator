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

La instalacion usa modo editable para que los cambios en `src/tfm_discord_moderator/` se reflejen sin reinstalar el paquete.

## 2. Configurar llama.cpp

El cliente espera un servidor local compatible con OpenAI Chat Completions:

```text
http://127.0.0.1:8001/v1/chat/completions
```

Ajusta `.env` si usas otro puerto, alias de modelo o timeout:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=discord-qwen-local
LLM_TIMEOUT_SECONDS=120
# Reservado para pruebas controladas futuras. Mantener en false.
AUTO_DELETE=false
```

Ejemplo de arranque en Windows. Cambia las rutas a tu instalacion real de `llama.cpp` y al modelo GGUF local:

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

No guardes `.env`, tokens, modelos GGUF ni datasets privados dentro del repositorio.

## 3. Verificar

Con el entorno Python activado:

```powershell
python scripts/check_environment.py
pytest
python scripts/check_llama_server.py
python scripts/classify_sample.py
```

`check_llama_server.py` y `classify_sample.py` requieren que `llama-server` este levantado.

## 4. Bot de Discord experimental

El adaptador de Discord esta pensado para observacion y pruebas controladas. No ejecuta acciones automaticas de moderacion en esta fase.

1. Crea una aplicacion/bot en el portal de Discord.
2. Activa el intent de contenido de mensajes para tu bot de pruebas.
3. Copia el token en `.env` como `DISCORD_TOKEN`.
4. Manten `AUTO_DELETE=false`.

```powershell
python -m tfm_discord_moderator.discord_bot.bot
```

## 5. Siguientes pasos

- Definir el formato JSONL normalizado de datos.
- Preparar un conjunto de evaluacion pequeno, anonimizado y versionable si procede.
- Medir latencia, precision, falsos positivos y falsos negativos.
- Revisar la taxonomia inicial antes de convertirla en definitiva.
