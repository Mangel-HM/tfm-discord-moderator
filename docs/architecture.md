# Arquitectura inicial

Flujo principal:

```text
.env/.env.example -> config

Discord API -> discord_bot -> domain schemas -> classification -> baseline_classifier -> inference -> llama.cpp server -> Qwen GGUF
                                                |              -> lora_classifier -> Transformers + PEFT -> Qwen + LoRA adapter
                                                |
                                                v
                                           evaluation/logs

data/samples -> scripts de comprobacion -> classification/evaluation
```

Capas:

1. `domain`: modelos de datos independientes de Discord y del LLM.
2. `config`: carga de variables de entorno y valores seguros por defecto.
3. `inference`: cliente HTTP para el servidor local de llama.cpp.
4. `classification`: prompts, backends de clasificacion y parseo/validacion JSON.
5. `data/`: muestras sinteticas pequenas y, mas adelante, datos normalizados.
6. `evaluation`: metricas para comparar baseline y futuros modelos ajustados.
7. `discord_bot`: adaptador de entrada/salida para Discord, sin acciones automaticas de moderacion.

Decision importante: el bot no debe borrar mensajes por defecto. Primero se mide la
calidad del clasificador y solo despues se activan acciones automaticas en un servidor
controlado.

Para la demo LoRA, el bot usa Transformers + PEFT directamente y no exporta el adapter a
GGUF ni lo sirve mediante llama.cpp.

Contrato interno principal:

- `topic`
- `risk_labels`
- `action`
- opcionalmente `confidence`
- opcionalmente `rationale`

Los IDs internos de topics, risk labels y actions se conservan en prompts, JSONL,
predicciones, metricas y validacion. La presentacion humana puede mapear esos IDs a nombres
legibles en ingles, pero solo en avisos del canal de moderacion y salida por consola.

La ruta legacy basada en `label`, `risk`, `ClassificationResult` y `configs/labels.yml` ya
no forma parte de la arquitectura activa.
