# Datos

Estructura:

- `data/raw/`: datos originales, no versionar si contienen informacion sensible.
- `data/processed/`: datos limpios y anonimizados listos para evaluacion o entrenamiento.
- `data/samples/`: ejemplos sinteticos pequenos que si pueden versionarse.

## Muestras sinteticas

`data/samples/messages_sample.jsonl` se usa como entrada pequena para `just sample`.
El script actual lee estos campos de cada registro:

- `message_id`
- `channel`
- `author_role`
- `context`
- `text`

Campos antiguos como `expected_label` o `expected_action`, si aparecen en muestras
sinteticas, son legado y no forman parte del contrato activo de clasificacion.

## Formato normalizado activo

Los datasets preparados para baseline, evaluacion y SFT usan el formato normalizado
descrito en `docs/data_format.md`:

```json
{"id":"synthetic-001","source_dataset":"synthetic_discord","text":"Necesito ayuda para configurar el bot.","topic":"soporte","risk_labels":["sin_riesgo"],"action":"allow","split":"train","original_labels":{},"metadata":{"language":"es"}}
```

Los datos reales originales y derivados deben permanecer fuera de Git, normalmente bajo
`data/raw/` y `data/processed/`.
