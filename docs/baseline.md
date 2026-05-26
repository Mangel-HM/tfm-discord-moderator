# Baseline de clasificacion JSONL

Este baseline clasifica archivos JSONL en el formato normalizado del proyecto usando
Qwen3.5-2B servido localmente con `llama.cpp`. Es una linea base por prompting: no entrena
LoRA ni modifica pesos del modelo.

La fase actual esta limitada a mensajes en ingles. La validacion en espanol queda fuera de
esta implementacion y se documentara como trabajo futuro.

## Servidor local

Arranca `llama-server` antes de ejecutar el baseline. Ejemplo en Windows:

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

El endpoint esperado es:

```text
http://127.0.0.1:8001/v1/chat/completions
```

## Ejecucion

Ejemplo con una muestra pequena de Jigsaw ya normalizada:

```powershell
just baseline data/processed/jigsaw_train_5000.jsonl outputs/baseline_jigsaw_sample.jsonl 25 --continue-on-error
```

Comando equivalente sin `just`:

```powershell
uv run python scripts/run_baseline.py `
  --input data/processed/jigsaw_train_5000.jsonl `
  --output outputs/baseline_jigsaw_sample.jsonl `
  --max-examples 25 `
  --continue-on-error
```

Opciones utiles:

- `--base-url`: cambia el endpoint base. Por defecto `http://127.0.0.1:8001/v1`.
- `--model`: cambia el alias del modelo. Por defecto `discord-qwen-local`.
- `--timeout`: timeout de cada peticion en segundos.
- `--continue-on-error`: guarda errores de inferencia o parseo como filas con `parse_error`.

## Qwen3.5 y modo pensamiento

El cliente envia `chat_template_kwargs={"enable_thinking": false}` al servidor compatible
con OpenAI. Qwen3.5 puede generar primero una traza de razonamiento en
`reasoning_content`; para esta tarea de clasificacion corta con JSON estricto, eso puede
agotar `max_tokens` antes de producir la respuesta final en `content`.

Desactivar el modo pensamiento hace que el modelo responda directamente en `content`.
Esta es una decision experimental del baseline para priorizar salidas JSON parseables,
menor latencia y comportamiento repetible. No implica entrenamiento ni cambios en los
pesos del modelo.

## Formato de predicciones

El archivo de salida es JSONL, una prediccion por linea. Cada registro conserva el texto y
las etiquetas gold del ejemplo normalizado, y anade la salida predicha:

```json
{"id":"jigsaw_000001","source_dataset":"jigsaw","text":"...","gold_topic":"otro","gold_risk_labels":["sin_riesgo"],"gold_action":"allow","pred_topic":"otro","pred_risk_labels":["sin_riesgo"],"pred_action":"allow","confidence":0.91,"rationale":"No moderation risk.","latency_ms":123.4,"raw_response":"{...}","parse_error":null}
```

Si el modelo no devuelve JSON valido o usa etiquetas fuera de la taxonomia, el proceso se
detiene por defecto. Con `--continue-on-error`, la fila se escribe con `parse_error` y sin
prediccion validada.

Este baseline servira como comparacion frente al futuro modelo adaptado con LoRA.
