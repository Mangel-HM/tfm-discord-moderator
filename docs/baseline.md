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

## Evaluacion automatica

Las predicciones del baseline se evaluan offline, sin volver a llamar al modelo:

```powershell
just evaluate-baseline outputs/baseline_jigsaw_sample.jsonl outputs/baseline_metrics.json --ignore-topic
```

Comando equivalente sin `just`:

```powershell
uv run python scripts/evaluate_baseline.py `
  --input outputs/baseline_jigsaw_sample.jsonl `
  --output outputs/baseline_metrics.json `
  --ignore-topic
```

El flag `--ignore-topic` es recomendable para Jigsaw porque el conversor asigna
`topic="otro"` de forma artificial. Ese dataset no contiene anotacion tematica real, por lo
que medir `topic_accuracy` en Jigsaw no aporta una comparacion valida.

El archivo de metricas se escribe como JSON legible. Las metricas de clasificacion se
calculan solo sobre predicciones parseadas correctamente; los fallos de JSON se miden aparte
con `parse_errors` y `valid_json_rate`.

Metricas principales:

- `total_examples`: numero total de filas validas leidas del JSONL de predicciones.
- `parsed_predictions`: predicciones con JSON valido y etiquetas aceptadas por la taxonomia.
- `parse_errors`: filas generadas con `parse_error`; indican fallos de inferencia, parseo o
  validacion de la salida del modelo.
- `valid_json_rate`: proporcion de filas parseadas correctamente. Es importante porque el
  baseline exige JSON valido antes de usar cualquier prediccion.
- `action_accuracy`: exactitud de la accion recomendada (`allow`, `review`,
  `warn_candidate`, `delete_candidate`). Resume si el sistema recomienda el mismo nivel de
  actuacion que la etiqueta gold.
- `risk_exact_match_accuracy`: proporcion de ejemplos donde el conjunto completo de
  `risk_labels` coincide exactamente con el gold. Trata las etiquetas como multi-label, por
  lo que el orden no importa.
- `risk_labels`: resumen por etiqueta con verdaderos positivos, falsos positivos, falsos
  negativos, `gold_support`, `predicted_support`, `precision`, `recall` y `f1`. Esto permite
  ver que tipos de riesgo detecta bien el baseline y cuales confunde u omite.
- `macro_f1`: media de F1 por etiquetas con soporte gold. Sirve para comparar modelos sin
  depender solo de las clases mas frecuentes.
- `average_latency_ms`, `min_latency_ms`, `max_latency_ms`: resumen de latencia por ejemplo,
  usando el campo `latency_ms` escrito por el baseline. Incluyen tambien filas con
  `parse_error`, porque esas filas tambien consumieron tiempo de inferencia.
- `topic_accuracy`: exactitud de `topic`, solo cuando no se usa `--ignore-topic`.

Los resultados deben interpretarse como una medicion de la fase actual en ingles. La
validacion en espanol queda fuera de esta evaluacion y se documentara mas adelante como
trabajo futuro.

Este baseline sirve como comparacion frente al adapter LoRA evaluado con el mismo contrato
normalizado.
