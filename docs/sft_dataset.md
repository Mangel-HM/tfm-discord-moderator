# Dataset SFT/chat

Este formato prepara ejemplos normalizados para el futuro entrenamiento supervisado de
LoRA. El script no entrena modelos, no llama a `llama.cpp` y no genera predicciones: solo
transforma etiquetas gold ya normalizadas en conversaciones de entrenamiento.

Cada linea del JSONL de salida contiene tres mensajes (`system`, `user`, `assistant`) y
metadatos mínimos para trazabilidad. El mensaje `assistant` es un JSON valido con `topic`,
`risk_labels` y `action`. No incluye `confidence`, porque esa puntuacion es una estimacion
del modelo durante inferencia, no una etiqueta humana.

Ejemplo:

```powershell
just build-sft data/processed/jigsaw_train_5000.jsonl data/processed/jigsaw_train_5000_sft.jsonl --exclude-ids-from data/processed/jigsaw_eval_balanced.jsonl --shuffle --seed 42
```

`--exclude-ids-from` debe usarse para evitar mezclar ejemplos de entrenamiento con muestras
de evaluacion o predicciones ya reservadas para medir el baseline. Puede repetirse y acepta
JSONL normalizado o JSONL de predicciones siempre que cada linea tenga un campo `id`.

Los datasets SFT generados son artefactos derivados de datos reales. Deben quedarse fuera
de Git, normalmente bajo `data/processed/`, y no deben subirse al repositorio.

La fase actual trabaja solo con datos en ingles. La validacion en espanol queda como
trabajo futuro.
