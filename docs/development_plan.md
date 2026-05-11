# Plan de segunda iteracion

## Objetivo

Validar que el entorno local ejecuta Qwen3.5-2B mediante llama.cpp y dejar preparado el repositorio para implementar el baseline por prompting y la futura evaluacion.

## Hitos

1. Comprobar GPU y entorno Windows.
2. Ejecutar Qwen3.5-2B-GGUF con llama-server.
3. Hacer una peticion HTTP compatible con OpenAI Chat Completions.
4. Crear repositorio GitHub con arquitectura base.
5. Implementar baseline de clasificacion con salida JSON validada.
6. Probar el baseline con ejemplos sinteticos.
7. Preparar dataset real/anonimizado para evaluacion posterior.

## Criterios de exito

- El servidor local responde en `http://127.0.0.1:8001/v1/chat/completions`.
- `python scripts/check_llama_server.py` devuelve una respuesta.
- `python scripts/classify_sample.py` clasifica los ejemplos JSONL sin romper el parseo.
- El bot puede conectarse a un servidor de Discord de pruebas sin borrar mensajes.
