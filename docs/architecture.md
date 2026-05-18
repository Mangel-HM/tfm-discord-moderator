# Arquitectura inicial

Flujo principal:

```text
.env/.env.example -> config

Discord API -> discord_bot -> domain schemas -> baseline_classifier -> inference -> llama.cpp server -> Qwen GGUF
                                  |                                             |
                                  v                                             v
                              evaluation                                  logs/resultados

data/samples -> scripts de comprobacion -> classification/evaluation
```

Capas:

1. `domain`: modelos de datos independientes de Discord y del LLM.
2. `config`: carga de variables de entorno y valores seguros por defecto.
3. `inference`: cliente HTTP para el servidor local de llama.cpp.
4. `classification`: prompts, taxonomia inicial y parseo/validacion JSON.
5. `data/`: muestras sinteticas pequenas y, mas adelante, datos normalizados.
6. `evaluation`: metricas para comparar baseline y futuros modelos ajustados.
7. `discord_bot`: adaptador de entrada/salida para Discord, sin acciones automaticas de moderacion.

Decision importante: en la segunda iteracion el bot no debe borrar mensajes por defecto. Primero se mide la calidad del clasificador y solo despues se activan acciones automaticas en un servidor controlado.
