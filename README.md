# TFM Discord Moderator

Proyecto base para una prueba de concepto de clasificacion tematica y apoyo a moderacion en Discord con un LLM local servido por llama.cpp.

## 1. Preparar entorno

Instala los prerrequisitos:

```powershell
winget install --id astral-sh.uv -e
winget install --id Casey.Just -e
uv --version
just --version
```

Despues, dentro del repositorio:

```powershell
just dev
```

El proyecto usa `uv` para crear y sincronizar `.venv` con Python 3.11. La instalacion es editable, por lo que los cambios en `src/tfm_discord_moderator/` se reflejan sin reinstalar el paquete.

`just dev` crea `.env` desde `.env.example` solo si no existe. Si ya tienes un `.env` local con tokens o rutas propias, no lo sobrescribe.

Recomendacion: instala `uv` y `just` como herramientas del sistema, no solo dentro de `.venv`.
Si `just` se ejecuta desde `.venv\Scripts\just.exe`, Windows puede bloquearlo mientras `uv sync`
actualiza el entorno. El grupo `dev` incluye `rust-just` y `uv` para evitar que `uv` elimine
esas herramientas durante la sincronizacion, pero la instalacion global sigue siendo el flujo mas limpio.

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

## 3. Recetas just

Las tareas habituales estan centralizadas en `justfile`:

```powershell
just --list
```

Recetas principales:

| Receta | Uso |
| --- | --- |
| `just dev` | Sincroniza el entorno, prepara `.env` si falta y muestra los siguientes comandos. |
| `just install` | Actualiza `uv.lock` y sincroniza dependencias con `uv`. |
| `just check` | Ejecuta lint, typecheck y tests. |
| `just lint` | Ejecuta `ruff check .`. |
| `just typecheck` | Ejecuta `mypy src scripts tests`. |
| `just test` | Ejecuta `pytest`. |
| `just env` | Muestra informacion local de Python, SO y GPU. |
| `just llama-check` | Comprueba el endpoint local de llama.cpp. |
| `just sample` | Clasifica los ejemplos sinteticos de `data/samples/messages_sample.jsonl`. |
| `just bot` | Ejecuta el bot de Discord en modo observacion. |

`uv.lock` se versiona para que el entorno sea reproducible. Como decision del proyecto, `just install` actualiza dependencias y despues sincroniza el entorno.

## 4. Verificar

Para comprobar tests, lint y tipos:

```powershell
just check
```

Para revisar Python, sistema operativo y GPU/CUDA:

```powershell
just env
```

Con `llama-server` levantado:

```powershell
just llama-check
just sample
```

`llama-check` y `sample` requieren que `llama-server` este levantado.

## 5. Bot de Discord experimental

El adaptador de Discord esta pensado para observacion y pruebas controladas. No ejecuta acciones automaticas de moderacion en esta fase.

1. Crea una aplicacion/bot en el portal de Discord.
2. Activa el intent de contenido de mensajes para tu bot de pruebas.
3. Copia el token en `.env` como `DISCORD_TOKEN`.
4. Manten `AUTO_DELETE=false`.

```powershell
just bot
```

## 6. Siguientes pasos

- Definir el formato JSONL normalizado de datos.
- Preparar un conjunto de evaluacion pequeno, anonimizado y versionable si procede.
- Medir latencia, precision, falsos positivos y falsos negativos.
- Revisar la taxonomia inicial antes de convertirla en definitiva.
