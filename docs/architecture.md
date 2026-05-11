# Arquitectura inicial

Flujo principal:

```text
Discord API -> discord_bot -> domain schemas -> baseline_classifier -> llama.cpp server -> Qwen GGUF
                                  |                                      |
                                  v                                      v
                              evaluacion                           logs/resultados
```

Capas:

1. `domain`: modelos de datos independientes de Discord y del LLM.
2. `inference`: cliente HTTP para el servidor local de llama.cpp.
3. `classification`: prompts, taxonomia y parseo/validacion JSON.
4. `discord_bot`: adaptador de entrada/salida para Discord.
5. `evaluation`: metricas para comparar baseline y futuros modelos ajustados.

Decision importante: en la segunda iteracion el bot no debe borrar mensajes por defecto. Primero se mide la calidad del clasificador y solo despues se activan acciones automaticas en un servidor controlado.
