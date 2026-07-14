# Dataset SFT/chat

Este formato prepara ejemplos normalizados para el entrenamiento supervisado de
LoRA. El script no entrena modelos, no llama a `llama.cpp` y no genera predicciones: solo
transforma etiquetas gold ya normalizadas en conversaciones de entrenamiento.

Cada linea del JSONL de salida contiene tres mensajes (`system`, `user`, `assistant`) y
metadatos mínimos para trazabilidad. El mensaje `assistant` es un JSON valido con `topic`,
`risk_labels` y `action`. No incluye `confidence`, porque esa puntuacion es una estimacion
del modelo durante inferencia, no una etiqueta humana.

Ejemplo:

```powershell
just build-sft data/processed/<train_normalized>.jsonl data/processed/<train_sft>.jsonl --exclude-ids-from data/processed/<eval_reserved>.jsonl --shuffle --seed 42
```

`--exclude-ids-from` debe usarse para evitar mezclar ejemplos de entrenamiento con muestras
de evaluacion o predicciones ya reservadas para medir el baseline. Puede repetirse y acepta
JSONL normalizado o JSONL de predicciones siempre que cada linea tenga un campo `id`.

Los datasets SFT generados son artefactos derivados de datos reales. Deben quedarse fuera
de Git, normalmente bajo `data/processed/`, y no deben subirse al repositorio.

La fase actual trabaja solo con datos en ingles. La validacion en espanol queda como
trabajo futuro.

## Artefacto Jigsaw balanced v2

`data/processed/jigsaw_train_balanced_v2_sft.jsonl` es el dataset SFT/chat construido a
partir del pool balanceado v2. Contiene 4245 conversaciones de entrenamiento con mensajes
`system`, `user` y `assistant`; la respuesta del assistant es JSON valido con `topic`,
`risk_labels` y `action`.

Se uso como entrada de entrenamiento para `outputs/lora_jigsaw_balanced_v2`. Como el resto
de datasets derivados, debe permanecer fuera de Git y conservarse solo como artefacto local
o copia de seguridad.
