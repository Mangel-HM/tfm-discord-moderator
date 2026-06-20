# LoRA

Este documento describe la ruta LoRA del proyecto: una comprobacion minima de entrenamiento,
la inferencia del adapter sobre JSONL normalizado y la evaluacion balanceada v2. La primera
parte sigue siendo un smoke test tecnico; los resultados v2 documentan una ejecucion
experimental concreta, no un ajuste exhaustivo de hiperparametros.

La linea base del proyecto se mantiene en `llama.cpp` con modelos GGUF. Para esta prueba,
LoRA se entrena con Transformers, PEFT y TRL, y la prueba preliminar del adapter tambien
se hace con Transformers y PEFT.

Queda fuera de esta tarea exportar el adapter a GGUF o servir un modelo LoRA con
`llama.cpp`. Eso se tratara como trabajo futuro si hace falta compararlo con la linea
base local.

Los adapters, checkpoints, modelos y outputs son artefactos generados. No deben subirse a
Git. Usa rutas ignoradas como `outputs/`.

## Entrenamiento smoke test

Ejemplo:

```powershell
just train-lora data/processed/<train_sft>.jsonl outputs/lora_smoke --model-name-or-path <MODELO_BASE_TRANSFORMERS> --max-examples 100 --max-steps 20 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --bf16 --gradient-checkpointing
```

El script:

- lee un JSONL SFT/chat con columna `messages`;
- carga tokenizer y modelo base con Transformers;
- aplica LoRA con PEFT;
- entrena unos pocos pasos con `SFTTrainer`;
- guarda adapter, tokenizer y `training_summary.json` en `output-dir`.

El modelo/tokenizer pasado por `--model-name-or-path` debe ser compatible con
Transformers y tener `chat_template`.

## Probar el adapter

Ejemplo:

```powershell
just test-lora-adapter outputs/lora_smoke --model-name-or-path <MODELO_BASE_TRANSFORMERS> --bf16
```

El script carga el modelo base, aplica el adapter LoRA y genera una respuesta corta para un
prompt de clasificacion compatible con las etiquetas del proyecto. En esta fase no parsea
ni evalua la salida.

## Inferencia y evaluacion sobre JSONL normalizado

Para evaluar un adapter ya entrenado sobre el conjunto normalizado reservado:

```powershell
just lora-adapter data/processed/<eval_normalized>.jsonl outputs/<lora_predictions>.jsonl outputs/<adapter_dir> --model-name-or-path Qwen/Qwen3.5-2B --bf16 --continue-on-error
```

El script usa el modelo base de Transformers, carga encima el adapter con PEFT y escribe un
JSONL compatible con el evaluador del baseline. La respuesta esperada del adapter sigue el
formato SFT de tres campos: `topic`, `risk_labels` y `action`. Si el modelo no devuelve JSON
valido o usa valores fuera de la taxonomia, con `--continue-on-error` la fila se conserva
con `parse_error`.

La evaluacion se hace sin volver a llamar al modelo:

```powershell
just evaluate-baseline outputs/<lora_predictions>.jsonl outputs/<lora_metrics>.json --ignore-topic
```

Para comparar contra una linea base justa, recalcula tambien el baseline con la taxonomia
actual y el mismo conjunto de evaluacion:

```powershell
just baseline data/processed/<eval_normalized>.jsonl outputs/<baseline_predictions>.jsonl --continue-on-error

just evaluate-baseline outputs/<baseline_predictions>.jsonl outputs/<baseline_metrics>.json --ignore-topic
```

`topic` se mantiene como campo de salida para conservar el contrato del proyecto, pero en
Jigsaw no debe tratarse como la metrica experimental principal. La comparacion principal
debe centrarse en `risk_labels`, `action`, `macro_f1`, `risk_exact_match_accuracy` y
`valid_json_rate`.

## Resultados Jigsaw balanced v2

La ejecucion `outputs/lora_jigsaw_balanced_v2` entreno un adapter sobre
`data/processed/jigsaw_train_balanced_v2_sft.jsonl` con `Qwen/Qwen3.5-2B`, LoRA `r=8`,
`alpha=16`, `dropout=0.05`, `bf16`, batch por dispositivo 1, acumulacion 4 y 800 pasos.
El `training_summary.json` registra 4245 ejemplos usados, perdida final aproximada
`0.7353`, 1887M parametros totales y 5.46M parametros entrenables.

La comparacion sobre `data/processed/jigsaw_eval_balanced_v2.jsonl` usa 302 ejemplos y
`--ignore-topic`, porque Jigsaw conserva `topic="otro"`. Los resumenes principales son:

- `outputs/lora_jigsaw_balanced_v2_metrics.json`: `macro_f1=0.7967`,
  `risk_exact_match_accuracy=0.5033`, `action_accuracy=0.9106`,
  `valid_json_rate=1.0`, sin errores de parseo.
- `outputs/baseline_jigsaw_eval_balanced_v2_metrics.json`: `macro_f1=0.5069`,
  `risk_exact_match_accuracy=0.4172`, `action_accuracy=0.7252`,
  `valid_json_rate=1.0`, sin errores de parseo.

Estos JSON de metricas y el directorio del adapter son artefactos generados. Deben quedarse
fuera de Git y conservarse, si hace falta, mediante copias externas.

## Troubleshooting

Si aparece CUDA out of memory:

- reduce `--max-seq-length`;
- manten `--per-device-train-batch-size 1`;
- usa `--gradient-checkpointing`;
- reduce `--max-examples`;
- prueba `--fp16` si `--bf16` falla, o al reves;
- revisa `TARGET_MODULES` en `scripts/train_lora.py` si PEFT indica que no encuentra
  modulos para aplicar LoRA.
