# Datasets

## Jigsaw Toxic Comment Classification Challenge

Jigsaw es el primer dataset publico usado en el proyecto para preparar ejemplos de riesgo
moderativo. El CSV original debe descargarse manualmente y colocarse fuera de Git, por
ejemplo en `data/raw/jigsaw/`. No se deben subir al repositorio ni el CSV original ni los
JSONL generados.

El conversor genera registros en el formato normalizado descrito en `docs/data_format.md`.
Como Jigsaw no incluye una tematica de Discord, todos los ejemplos se convierten con
`topic="otro"`.

Mapeo de riesgo:

- `toxic`, `severe_toxic` o `insult` -> `insulto_toxicidad`
- `identity_hate` -> `odio_discriminacion`
- `threat` -> `amenaza_violencia`
- `obscene` -> `sexual_nsfw`
- sin etiquetas activas -> `sin_riesgo`

Jigsaw no cubre `spam_fraude`, por lo que esa etiqueta no se genera desde este dataset.

Ejemplo de uso:

```powershell
just prepare-jigsaw data/raw/jigsaw/train.csv data/processed/jigsaw_train.jsonl train
```

### Receta just para Jigsaw

La receta del proyecto para ejecutar este conversor especifico de Jigsaw es:

```powershell
just prepare-jigsaw INPUT OUTPUT SPLIT
```

Argumentos:

- `INPUT`: ruta del CSV original de Jigsaw.
- `OUTPUT`: ruta del JSONL normalizado que se generara.
- `SPLIT`: particion asignada a los registros generados: `train`, `validation` o `test`.

Esta receta no es generica para cualquier dataset: asume las columnas y etiquetas del
Jigsaw Toxic Comment Classification Challenge. Si se incorporan otros datasets, conviene
crear recetas separadas como `prepare-civil-comments` o `prepare-toxicchat`, manteniendo
cada conversor simple y trazable.

Opciones utiles del script:

```powershell
uv run python scripts/prepare_jigsaw.py `
  --input data/raw/jigsaw/train.csv `
  --output data/processed/jigsaw_train.jsonl `
  --split train `
  --max-examples 100
```

Por defecto, el conversor falla si faltan columnas obligatorias o una fila no se puede
validar. Para omitir filas invalidas de forma explicita:

```powershell
uv run python scripts/prepare_jigsaw.py `
  --input data/raw/jigsaw/train.csv `
  --output data/processed/jigsaw_train.jsonl `
  --split train `
  --skip-invalid-rows
```
