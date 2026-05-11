# Datos

Estructura inicial:

- `data/raw/`: datos originales, no versionar si contienen informacion sensible.
- `data/processed/`: datos limpios y anonimizados listos para evaluacion o entrenamiento.
- `data/samples/`: ejemplos sinteticos pequenos que si pueden versionarse.

Formato JSONL para evaluacion:

```json
{"message_id":"ex-001","channel":"general","author_role":"member","context":["mensaje anterior"],"text":"mensaje a clasificar","expected_label":"soporte_tecnico","expected_action":"allow"}
```
