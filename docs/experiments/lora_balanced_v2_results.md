# Resultados LoRA balanceada v2

Este documento resume la evaluacion balanceada v2 del baseline por prompting frente a un
adapter LoRA entrenado sobre un SFT balanceado derivado de Jigsaw completo.

## Objetivo

El objetivo del experimento es comparar dos aproximaciones sobre la misma tarea de
clasificacion de moderacion:

- baseline con `Qwen/Qwen3.5-2B` mediante prompting estructurado;
- adapter LoRA sobre `Qwen/Qwen3.5-2B`, entrenado con PEFT/TRL sobre un SFT balanceado.

La comparacion se centra en `risk_labels`, `action`, `macro_f1`,
`risk_exact_match_accuracy` y `valid_json_rate`. El campo `topic` se mantiene como parte
del contrato estructural del proyecto, pero no se evalua como metrica principal en este
experimento.

## Dataset

La fuente de datos es Jigsaw Toxic Comment Classification, usando el `train.csv` completo.
El dataset normalizado contiene 159571 ejemplos.

Las etiquetas de riesgo usadas son:

- `sin_riesgo`
- `insulto_toxicidad`
- `odio_discriminacion`
- `amenaza_violencia`
- `sexual_nsfw`

La taxonomia activa del experimento se limita a las etiquetas anteriores. Como Jigsaw
procede de comentarios web y no incluye una taxonomia tematica de Discord, `topic` se
conserva como campo estructural, pero no se usa como metrica principal.

## Particion experimental

La evaluacion balanceada v2 contiene 302 ejemplos. Los soportes gold por etiqueta son:

| Etiqueta            | Ejemplos |
| ------------------- | -------: |
| sin_riesgo          |      100 |
| insulto_toxicidad   |      100 |
| sexual_nsfw         |      100 |
| odio_discriminacion |       92 |
| amenaza_violencia   |       51 |

El pool de entrenamiento balanceado v2 contiene 4398 ejemplos. Sus soportes por etiqueta
son:

| Etiqueta            | Ejemplos |
| ------------------- | -------: |
| sin_riesgo          |     2000 |
| insulto_toxicidad   |     2000 |
| sexual_nsfw         |     1756 |
| odio_discriminacion |      744 |
| amenaza_violencia   |      478 |

El SFT se genero excluyendo los ejemplos de evaluacion para evitar contaminacion entre
entrenamiento y test. El artefacto final `jigsaw_train_balanced_v2_sft.jsonl` contiene
4245 conversaciones de entrenamiento, tambien sin solapamiento de IDs con la evaluacion.
La diferencia frente al pool se debe a que el pool es el conjunto balanceado de partida,
mientras que el SFT final es el artefacto de entrenamiento efectivamente usado por TRL.

## Modelos comparados

- Baseline: `Qwen/Qwen3.5-2B` mediante prompting estructurado y servido con
  `llama.cpp/GGUF`.
- LoRA: `Qwen/Qwen3.5-2B` adaptado con PEFT/TRL sobre el SFT balanceado.

El entrenamiento LoRA uso `r=8`, `alpha=16`, `dropout=0.05`, `bf16`, batch por
dispositivo 1, acumulacion de gradiente 4 y 800 pasos. El resumen de entrenamiento registra
perdida final aproximada `0.7353`, 1887M parametros totales y 5.46M parametros entrenables.

## Resultados principales

| Metrica           | Baseline | LoRA balanceado v2 |
| ----------------- | -------: | -----------------: |
| Valid JSON rate   |    1.000 |              1.000 |
| Macro-F1          |    0.507 |              0.797 |
| Action accuracy   |    0.725 |              0.911 |
| Risk exact match  |    0.417 |              0.503 |
| Latencia media ms |      365 |               1546 |

## Resultados por etiqueta

| Etiqueta            | F1 baseline | F1 LoRA |
| ------------------- | ----------: | ------: |
| amenaza_violencia   |       0.455 |   0.800 |
| insulto_toxicidad   |       0.614 |   0.671 |
| odio_discriminacion |       0.663 |   0.824 |
| sexual_nsfw         |       0.077 |   0.821 |
| sin_riesgo          |       0.726 |   0.867 |

## Interpretacion

El adapter LoRA mejora claramente la calidad de clasificacion frente al baseline por
prompting. La mejora se observa en `macro_f1`, en la exactitud de accion y en todas las
etiquetas de riesgo medidas. El salto mas visible aparece en `sexual_nsfw`, donde el
baseline casi no recupera positivos, y en clases minoritarias como `amenaza_violencia` y
`odio_discriminacion`.

El coste principal es una latencia mayor. Esta comparacion no mide dos runtimes
equivalentes: el baseline se sirve mediante `llama.cpp` con GGUF, mientras que la evaluacion
LoRA se realiza con Transformers + PEFT. Por tanto, la mejora de calidad debe interpretarse
junto con el coste operativo de inferencia y con el trabajo pendiente de exportar o servir
el adapter en un entorno comparable.

## Limitaciones

- El dataset procede de comentarios web, no de mensajes reales de Discord.
- La evaluacion se realiza en ingles.
- `topic` no se evalua formalmente.
- El adapter LoRA no se ha exportado aun a GGUF.
- Las metricas se calculan sobre una muestra balanceada, no sobre la distribucion real de
  produccion.

## Artefactos contrastados

Los valores anteriores se contrastaron con los artefactos locales del experimento:

- `outputs/lora_jigsaw_balanced_v2/training_summary.json`
- `outputs/lora_jigsaw_balanced_v2_metrics.json`
- `outputs/baseline_jigsaw_eval_balanced_v2_metrics.json`
- `data/processed/jigsaw_eval_balanced_v2.jsonl`
- `data/processed/jigsaw_train_balanced_pool_v2.jsonl`
- `data/processed/jigsaw_train_balanced_v2_sft.jsonl`
