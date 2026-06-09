# Smoke test LoRA

Este documento describe una comprobacion minima de entrenamiento LoRA sobre el dataset
SFT/chat generado por el proyecto. No es un entrenamiento final, no ajusta
hiperparametros y no mide calidad de forma completa: solo comprueba que la ruta tecnica
funciona de extremo a extremo.

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
just train-lora data/processed/jigsaw_train_5000_sft.jsonl outputs/lora_smoke --model-name-or-path <MODELO_BASE_TRANSFORMERS> --max-examples 100 --max-steps 20 --per-device-train-batch-size 1 --gradient-accumulation-steps 4 --bf16 --gradient-checkpointing
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

## Troubleshooting

Si aparece CUDA out of memory:

- reduce `--max-seq-length`;
- manten `--per-device-train-batch-size 1`;
- usa `--gradient-checkpointing`;
- reduce `--max-examples`;
- prueba `--fp16` si `--bf16` falla, o al reves;
- revisa `TARGET_MODULES` en `scripts/train_lora.py` si PEFT indica que no encuentra
  modulos para aplicar LoRA.
