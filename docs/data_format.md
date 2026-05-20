# Formato normalizado de datos

Este formato define una estructura comun para representar mensajes usados en baseline,
evaluacion y futuros entrenamientos LoRA. Su objetivo es que datasets publicos, ejemplos
sinteticos y conjuntos de evaluacion puedan convertirse a una misma forma antes de usarse
en el proyecto.

## Por que JSONL

Se usa JSONL porque cada linea contiene un registro JSON independiente. Esto permite leer
archivos grandes de forma incremental, validar ejemplos uno a uno y combinar o filtrar
conjuntos sin cargar todo el dataset en memoria.

## Campos

- `id`: identificador unico del ejemplo dentro del conjunto convertido.
- `source_dataset`: nombre del dataset o fuente de origen.
- `text`: mensaje que se clasificara.
- `topic`: clasificacion tematica principal del mensaje.
- `risk_labels`: lista de etiquetas de riesgo moderativo.
- `action`: accion recomendada para apoyo a moderacion.
- `split`: particion del conjunto: entrenamiento, validacion o test.
- `original_labels`: etiquetas originales del dataset antes de normalizar.
- `metadata`: informacion auxiliar no usada como etiqueta principal.

Ejemplo valido:

```json
{"id":"synthetic-001","source_dataset":"synthetic_discord","text":"Necesito ayuda para configurar el bot.","topic":"soporte","risk_labels":["sin_riesgo"],"action":"allow","split":"train","original_labels":{"category":"help"},"metadata":{"language":"es"}}
```

## Topic, risk_labels y action

`topic` describe de que trata el mensaje. `risk_labels` describe si existe riesgo
moderativo y de que tipo. `action` no es una sancion automatica: es una recomendacion
conservadora para que el sistema pueda priorizar revision humana.

## Valores permitidos

Topics:

- `gaming`
- `soporte`
- `social_general`
- `otro`

Risk labels:

- `sin_riesgo`
- `insulto_toxicidad`
- `odio_discriminacion`
- `amenaza_violencia`
- `sexual_nsfw`
- `spam_fraude`

Actions:

- `allow`
- `review`
- `warn_candidate`
- `delete_candidate`

Splits:

- `train`
- `validation`
- `test`

Si `risk_labels` contiene `sin_riesgo`, no debe combinarse con ninguna otra etiqueta de
riesgo.

## Conversion de datasets

Los datasets reales se convertiran automaticamente mediante scripts futuros. No se deben
normalizar manualmente registros reales ni guardar datasets privados dentro del repositorio.
